SHELL := /bin/bash

APPTAINER ?= apptainer
XFOIL_SIF ?= bin/xfoil-ubuntu22.sif
XFOIL_DEF ?= bin/containers/xfoil-ubuntu22.def

.PHONY: help apptainer-check xfoil-apptainer-build xfoil-apptainer-check xfoil-apptainer-clean

help:
	@echo "Targets:"
	@echo "  make apptainer-check         # Verify apptainer is available in PATH"
	@echo "  make xfoil-apptainer-build   # Build Ubuntu22.04 XFOIL image at $(XFOIL_SIF)"
	@echo "  make xfoil-apptainer-check   # Verify xfoil runs inside image"
	@echo "  make xfoil-apptainer-clean   # Remove built image"

apptainer-check:
	@command -v $(APPTAINER) >/dev/null 2>&1 || { echo "apptainer not found in PATH"; exit 1; }
	@$(APPTAINER) --version

xfoil-apptainer-build:
	mkdir -p $(dir $(XFOIL_SIF))
	$(APPTAINER) build $(XFOIL_SIF) $(XFOIL_DEF)

xfoil-apptainer-check:
	$(APPTAINER) exec $(XFOIL_SIF) xfoil <<<'QUIT'

xfoil-apptainer-clean:
	rm -f $(XFOIL_SIF)
