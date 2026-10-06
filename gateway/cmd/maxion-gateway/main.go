// Command maxion-gateway runs the MaxionBench AI gateway.
package main

import (
	"context"
	"errors"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"

	"github.com/htang7415/MaxionBench/gateway/internal/budget"
	"github.com/htang7415/MaxionBench/gateway/internal/config"
	"github.com/htang7415/MaxionBench/gateway/internal/proxy"
)

func main() {
	cfgPath := flag.String("config", "gateway.yaml", "path to the gateway config")
	flag.Parse()
	log := slog.New(slog.NewJSONHandler(os.Stderr, nil))

	cfg, err := config.Load(*cfgPath)
	if err != nil {
		log.Error("config", "err", err.Error())
		os.Exit(2)
	}
	var key string
	var ledger *budget.Ledger
	if cfg.Remote.Enabled {
		if key, err = config.LoadKey(cfg.Remote); err != nil {
			log.Error("remote key", "err", err.Error()) // message never includes the key
			os.Exit(2)
		}
		if ledger, err = budget.Open(cfg.Budget.LedgerPath, cfg.Remote.CapUSD); err != nil {
			log.Error("ledger", "err", err.Error())
			os.Exit(2)
		}
	}
	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	gw := proxy.New(cfg, key, ledger, reg, log)
	srv := &http.Server{Addr: cfg.Listen, Handler: gw.Handler(reg), ReadHeaderTimeout: 10 * time.Second}

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()
	go func() {
		log.Info("listening", "addr", cfg.Listen, "policy", cfg.Policy, "remote", cfg.Remote.Enabled,
			"local_upstreams", len(cfg.Local.Upstreams))
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("serve", "err", err.Error())
			os.Exit(1)
		}
	}()
	<-ctx.Done()
	shutdown, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	srv.Shutdown(shutdown) //nolint:errcheck
}
