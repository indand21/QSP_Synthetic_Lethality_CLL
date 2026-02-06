#!/usr/bin/env Rscript
# Install required R packages for the Synthetic Lethality QSP project
#
# Usage:
#   Rscript install.R

packages <- c(
  "tidyverse",     # dplyr, ggplot2, tidyr, readr, purrr, stringr, forcats, tibble
  "ggplot2",       # Publication-quality plotting
  "broom",         # Tidy model outputs
  "car",           # ANOVA, Levene's test
  "pROC",          # ROC curve analysis
  "boot",          # Bootstrap methods
  "jsonlite",      # JSON I/O for Python integration
  "gridExtra",     # Multi-panel figure layouts
  "scales",        # Axis formatting
  "viridis",       # Colorblind-friendly palettes
  "RColorBrewer",  # Additional color palettes
  "knitr",         # Report generation
  "rmarkdown"      # R Markdown documents
)

# Install missing packages
installed <- installed.packages()[, "Package"]
to_install <- packages[!packages %in% installed]

if (length(to_install) > 0) {
  cat("Installing", length(to_install), "packages:\n")
  cat(paste(" -", to_install, collapse = "\n"), "\n\n")
  install.packages(to_install, repos = "https://cloud.r-project.org")
} else {
  cat("All required R packages are already installed.\n")
}

# Verify installation
cat("\nVerifying package installation:\n")
for (pkg in packages) {
  status <- if (requireNamespace(pkg, quietly = TRUE)) "OK" else "MISSING"
  cat(sprintf("  [%s] %s\n", status, pkg))
}

cat("\nR package installation complete.\n")
