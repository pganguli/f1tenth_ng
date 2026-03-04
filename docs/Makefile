# Minimal Makefile for Sphinx documentation

SPHINXOPTS ?=
SPHINXBUILD ?= sphinx-build
SOURCEDIR = source
BUILDDIR = build

# Packages to generate API docs for
ROOTDIR := $(dir $(realpath $(lastword $(MAKEFILE_LIST))))
PACKAGES = \
	$(ROOTDIR)/../packages/f110_gym/src/f110_gym \
	$(ROOTDIR)/../packages/f110_planning/src/f110_planning \
	$(ROOTDIR)/../packages/f110_scripts/src/f110_scripts

APIDIR = $(SOURCEDIR)/api

# Default target
.PHONY: help
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate API docs
.PHONY: apidoc
apidoc:
	@echo "Generating API docs..."
	@rm -rf $(APIDIR)
	@mkdir -p $(APIDIR)
	@for pkg in $(PACKAGES); do \
		pkg_name=$$(basename $$pkg); \
		echo "  -> $$pkg_name"; \
		sphinx-apidoc -T -o $(APIDIR) $$pkg $$pkg/envs/base_classes.py $$pkg/base_classes.py $$pkg/base.py; \
	done
	@echo "API docs generated."

# Generic Sphinx target (runs apidoc first)
.PHONY: html markdown latex epub
html markdown latex epub: apidoc
	@$(SPHINXBUILD) -b $@ "$(SOURCEDIR)" "$(BUILDDIR)/$@" $(SPHINXOPTS)

# Catch-all: route unknown targets to Sphinx "make mode"
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Clean build directory
.PHONY: clean
clean:
	@echo "Cleaning $(BUILDDIR)..."
	@rm -rf $(BUILDDIR)/*
