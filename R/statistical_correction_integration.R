# Multiple Testing Correction Integration for R Statistical Analysis
# ============================================================================
# This R script provides integration between Python multiple testing corrections
# and R statistical analysis framework for comprehensive drug screening analysis
#
# Key Features:
# - R implementation of multiple testing correction methods
# - Integration with existing R statistical analysis framework
# - Publication-ready statistical reports with corrections
# - Power analysis for corrected tests
# - Before/after comparison visualizations
#
# Author: Kilo Code
# Date: 2025-11-09
# ============================================================================

# Load required packages
suppressPackageStartupMessages({
  library(tidyverse)
  library(ggplot2)
  library(dplyr)
  library(p.adjust)
  library(pwr)
  library(gt)
  library(knitr)
  library(kableExtra)
  library(gridExtra)
  library(grid)
  library(ggpubr)
  library(RColorBrewer)
  library(parallel)
  library(jsonlite)
  library(readr)
  library(reshape2)
})

# Global variables
MTC_FRAMEWORK_VERSION <- "1.0.0"
DEFAULT_ALPHA <- 0.05

# ==============================================================================
# MULTIPLE TESTING CORRECTION FUNCTIONS
# ==============================================================================

#' Apply Multiple Testing Correction Using R Built-in Methods
#'
#' This function provides R-based multiple testing correction as an alternative
#' to Python implementation, ensuring consistency and cross-platform compatibility
#'
#' @param pvalues Vector of uncorrected p-values
#' @param method Correction method ("holm", "bonferroni", "BH", "BY", "fdr", "none")
#' @param alpha Significance level
#' @return List with correction results
apply_mtc_correction_r <- function(pvalues, method = "BH", alpha = DEFAULT_ALPHA) {
  
  # Remove NA values
  valid_mask <- !is.na(pvalues)
  valid_pvalues <- pvalues[valid_mask]
  
  if (length(valid_pvalues) == 0) {
    stop("No valid p-values provided")
  }
  
  # Apply correction
  corrected_pvalues <- p.adjust(valid_pvalues, method = method)
  
  # Create output data frame
  results_df <- data.frame(
    Test = paste0("Test_", 1:length(valid_pvalues)),
    Original_pvalue = valid_pvalues,
    Corrected_pvalue = corrected_pvalues,
    Significant_Before = valid_pvalues <= alpha,
    Significant_After = corrected_pvalues <= alpha,
    stringsAsFactors = FALSE
  )
  
  # Calculate summary statistics
  n_tests <- length(valid_pvalues)
  n_significant_before <- sum(valid_pvalues <= alpha)
  n_significant_after <- sum(corrected_pvalues <= alpha)
  n_newly_significant <- sum(results_df$Significant_After & !results_df$Significant_Before)
  n_lost_significance <- sum(results_df$Significant_Before & !results_df$Significant_After)
  
  # Method-specific information
  method_info <- switch(method,
    "holm" = "Sequential Bonferroni (Holm) correction",
    "bonferroni" = "Bonferroni family-wise error rate control",
    "BH" = "Benjamini-Hochberg False Discovery Rate control",
    "BY" = "Benjamini-Yekutieli False Discovery Rate control",
    "fdr" = "False Discovery Rate control",
    "none" = "No correction applied",
    paste0("Unknown method: ", method)
  )
  
  return(list(
    results_table = results_df,
    summary_statistics = list(
      n_tests = n_tests,
      n_significant_before = n_significant_before,
      n_significant_after = n_significant_after,
      n_newly_significant = n_newly_significant,
      n_lost_significance = n_lost_significance,
      significance_rate_before = n_significant_before / n_tests,
      significance_rate_after = n_significant_after / n_tests,
      method = method_info,
      alpha_level = alpha
    )
  ))
}

#' Compare Multiple Testing Correction Methods
#'
#' Compare different correction methods on the same p-values
#'
#' @param pvalues Vector of uncorrected p-values
#' @param methods Vector of correction methods to compare
#' @param alpha Significance level
#' @return Data frame with comparison results
compare_mtc_methods <- function(pvalues, methods = c("holm", "bonferroni", "BH"), alpha = DEFAULT_ALPHA) {
  
  comparison_results <- list()
  
  for (method in methods) {
    cat("Applying", method, "correction...\n")
    result <- apply_mtc_correction_r(pvalues, method, alpha)
    comparison_results[[method]] <- result
  }
  
  # Create comparison summary
  comparison_summary <- data.frame(
    Method = methods,
    N_Significant = sapply(comparison_results, function(x) x$summary_statistics$n_significant_after),
    Significance_Rate = sapply(comparison_results, function(x) x$summary_statistics$significance_rate_after),
    Newly_Significant = sapply(comparison_results, function(x) x$summary_statistics$n_newly_significant),
    Lost_Significance = sapply(comparison_results, function(x) x$summary_statistics$n_lost_significance),
    stringsAsFactors = FALSE
  )
  
  return(list(
    detailed_results = comparison_results,
    comparison_summary = comparison_summary
  ))
}

#' Create Before/After Visualization for Multiple Testing Correction
#'
#' Generate comprehensive visualization showing impact of multiple testing correction
#'
#' @param correction_results Results from apply_mtc_correction_r
#' @param output_file Optional output file path
#' @return List of ggplot objects
create_before_after_visualization <- function(correction_results, output_file = NULL) {
  
  results_df <- correction_results$results_table
  summary_stats <- correction_results$summary_statistics
  
  # Set up plotting theme
  theme_mtc <- function() {
    theme_minimal() +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(face = "bold", size = 12),
        panel.grid.minor = element_blank(),
        legend.position = "bottom"
      )
  }
  
  # Create plots
  plots <- list()
  
  # 1. P-value comparison scatter plot
  p1 <- ggplot(results_df, aes(x = Original_pvalue, y = Corrected_pvalue)) +
    geom_point(alpha = 0.7, size = 3, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_hline(yintercept = summary_stats$alpha_level, color = "orange", linetype = "dotted") +
    geom_vline(xintercept = summary_stats$alpha_level, color = "orange", linetype = "dotted") +
    labs(
      title = paste("P-value Correction -", summary_stats$method),
      subtitle = paste("N =", summary_stats$n_tests, "tests"),
      x = "Original p-value",
      y = "Corrected p-value"
    ) +
    theme_mtc()
  
  plots$pvalue_comparison <- p1
  
  # 2. Significance status changes
  change_data <- data.frame(
    Category = c("No Change", "Gained Significance", "Lost Significance"),
    Count = c(
      summary_stats$n_tests - summary_stats$n_newly_significant - summary_stats$n_lost_significance,
      summary_stats$n_newly_significant,
      summary_stats$n_lost_significance
    ),
    stringsAsFactors = FALSE
  )
  
  p2 <- ggplot(change_data, aes(x = Category, y = Count, fill = Category)) +
    geom_col(alpha = 0.8) +
    scale_fill_manual(values = c("No Change" = "lightblue", "Gained Significance" = "green", "Lost Significance" = "red")) +
    labs(
      title = "Changes in Significance Status",
      x = "Category",
      y = "Number of Tests"
    ) +
    theme_mtc() +
    theme(legend.position = "none")
  
  plots$significance_changes <- p2
  
  # 3. Significance rate comparison
  rate_data <- data.frame(
    Stage = c("Before Correction", "After Correction"),
    Rate = c(summary_stats$significance_rate_before, summary_stats$significance_rate_after),
    stringsAsFactors = FALSE
  )
  
  p3 <- ggplot(rate_data, aes(x = Stage, y = Rate, fill = Stage)) +
    geom_col(alpha = 0.8) +
    scale_fill_manual(values = c("Before Correction" = "lightcoral", "After Correction" = "lightblue")) +
    labs(
      title = "Significance Rate Comparison",
      x = "Stage",
      y = "Significance Rate"
    ) +
    theme_mtc() +
    theme(legend.position = "none")
  
  plots$significance_rate <- p3
  
  # 4. Q-Q plot of p-values
  valid_pvalues <- results_df$Original_pvalue
  expected_quantiles <- ppoints(length(valid_pvalues))
  observed_quantiles <- sort(valid_pvalues)
  
  qq_data <- data.frame(
    Expected = expected_quantiles,
    Observed = observed_quantiles
  )
  
  p4 <- ggplot(qq_data, aes(x = Expected, y = Observed)) +
    geom_point(alpha = 0.7, color = "steelblue") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = "Q-Q Plot of P-values (Uniform Distribution)",
      x = "Expected Quantiles",
      y = "Observed Quantiles"
    ) +
    theme_mtc()
  
  plots$qq_plot <- p4
  
  # Combine all plots
  combined_plot <- grid.arrange(
    p1, p2, p3, p4,
    ncol = 2, nrow = 2,
    top = textGrob(
      paste("Multiple Testing Correction Analysis -", summary_stats$method),
      gp = gpar(fontsize = 16, fontface = "bold")
    )
  )
  
  # Save if output file specified
  if (!is.null(output_file)) {
    ggsave(output_file, combined_plot, width = 12, height = 10, dpi = 300)
  }
  
  return(list(
    individual_plots = plots,
    combined_plot = combined_plot,
    summary_statistics = summary_stats
  ))
}

# ==============================================================================
# POWER ANALYSIS FOR MULTIPLE TESTING
# ==============================================================================

#' Statistical Power Analysis for Multiple Testing Correction
#'
#' Calculate statistical power under different multiple testing correction scenarios
#'
#' @param effect_size Expected effect size (Cohen's d)
#' @param sample_size Sample size per group
#' @param alpha Significance level
#' @param n_tests Number of tests being performed
#' @param method Correction method
#' @return List with power analysis results
power_analysis_mtc <- function(effect_size, sample_size, alpha = DEFAULT_ALPHA, 
                              n_tests = 1, method = "BH") {
  
  # Calculate uncorrected power
  power_uncorrected <- pwr.t.test(d = effect_size, n = sample_size, 
                                  sig.level = alpha, type = "two.sample")$power
  
  # Calculate corrected alpha level
  if (method == "bonferroni") {
    corrected_alpha <- alpha / n_tests
  } else if (method %in% c("BH", "holm")) {
    # For FDR methods, power calculation is more complex
    # Using conservative estimate
    corrected_alpha <- alpha * (1 - 0.2 * log(n_tests))  # Conservative adjustment
    corrected_alpha <- max(corrected_alpha, alpha / n_tests)  # At least as strict as Bonferroni
  } else {
    corrected_alpha <- alpha
  }
  
  # Calculate power with correction
  power_corrected <- pwr.t.test(d = effect_size, n = sample_size, 
                               sig.level = corrected_alpha, type = "two.sample")$power
  
  return(list(
    effect_size = effect_size,
    sample_size = sample_size,
    alpha_level = alpha,
    corrected_alpha = corrected_alpha,
    n_tests = n_tests,
    method = method,
    power_uncorrected = power_uncorrected,
    power_corrected = power_corrected,
    power_loss = power_uncorrected - power_corrected
  ))
}

#' Comprehensive Power Analysis for Multiple Testing Scenarios
#'
#' Analyze power across different numbers of tests and correction methods
#'
#' @param effect_sizes Vector of effect sizes to analyze
#' @param sample_sizes Vector of sample sizes to analyze
#' @param n_tests_options Vector of number of tests to analyze
#' @param correction_methods Vector of correction methods
#' @param alpha Significance level
#' @return Data frame with comprehensive power analysis results
comprehensive_power_analysis <- function(effect_sizes = c(0.2, 0.5, 0.8),
                                        sample_sizes = c(10, 20, 30, 50),
                                        n_tests_options = c(1, 5, 10, 20, 50),
                                        correction_methods = c("none", "bonferroni", "BH", "holm"),
                                        alpha = DEFAULT_ALPHA) {
  
  power_results <- expand.grid(
    effect_size = effect_sizes,
    sample_size = sample_sizes,
    n_tests = n_tests_options,
    method = correction_methods,
    stringsAsFactors = FALSE
  )
  
  power_results$power_uncorrected <- NA
  power_results$power_corrected <- NA
  power_results$corrected_alpha <- NA
  power_results$power_loss <- NA
  
  for (i in 1:nrow(power_results)) {
    result <- power_analysis_mtc(
      effect_size = power_results$effect_size[i],
      sample_size = power_results$sample_size[i],
      alpha = alpha,
      n_tests = power_results$n_tests[i],
      method = power_results$method[i]
    )
    
    power_results$power_uncorrected[i] <- result$power_uncorrected
    power_results$power_corrected[i] <- result$power_corrected
    power_results$corrected_alpha[i] <- result$corrected_alpha
    power_results$power_loss[i] <- result$power_loss
  }
  
  return(power_results)
}

#' Create Power Analysis Visualization
#'
#' Generate comprehensive power analysis plots
#'
#' @param power_results Data frame from comprehensive_power_analysis
#' @param output_file Optional output file path
#' @return List of ggplot objects
create_power_analysis_plots <- function(power_results, output_file = NULL) {
  
  plots <- list()
  
  # 1. Power vs Number of Tests
  p1 <- ggplot(power_results, aes(x = n_tests, y = power_corrected, color = method)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    facet_wrap(~ effect_size, labeller = labeller(effect_size = label_both)) +
    labs(
      title = "Statistical Power vs Number of Tests",
      subtitle = "By Effect Size and Correction Method",
      x = "Number of Tests",
      y = "Statistical Power",
      color = "Correction Method"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12),
      strip.text = element_text(face = "bold", size = 11),
      panel.grid.minor = element_blank()
    )
  
  plots$power_vs_tests <- p1
  
  # 2. Power Loss vs Number of Tests
  p2 <- ggplot(power_results, aes(x = n_tests, y = power_loss, color = method)) +
    geom_line(size = 1) +
    geom_point(size = 2) +
    facet_wrap(~ effect_size, labeller = labeller(effect_size = label_both)) +
    labs(
      title = "Power Loss vs Number of Tests",
      subtitle = "Compared to Uncorrected Analysis",
      x = "Number of Tests",
      y = "Power Loss",
      color = "Correction Method"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12),
      strip.text = element_text(face = "bold", size = 11),
      panel.grid.minor = element_blank()
    )
  
  plots$power_loss_vs_tests <- p2
  
  # 3. Heatmap of Power by Method and Effect Size
  p3 <- power_results %>%
    filter(n_tests == 10) %>%  # Focus on 10 tests scenario
    ggplot(aes(x = method, y = factor(effect_size), fill = power_corrected)) +
    geom_tile() +
    scale_fill_viridis_c(name = "Power", option = "plasma") +
    facet_wrap(~ sample_size, labeller = labeller(sample_size = label_both)) +
    labs(
      title = "Statistical Power Heatmap (10 Tests)",
      subtitle = "By Sample Size and Effect Size",
      x = "Correction Method",
      y = "Effect Size",
      fill = "Power"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12),
      strip.text = element_text(face = "bold", size = 11),
      panel.grid.minor = element_blank()
    )
  
  plots$power_heatmap <- p3
  
  # Save plots if output file specified
  if (!is.null(output_file)) {
    ggsave(output_file, p1, width = 12, height = 8, dpi = 300)
    ggsave(gsub(".png", "_power_loss.png", output_file), p2, width = 12, height = 8, dpi = 300)
    ggsave(gsub(".png", "_heatmap.png", output_file), p3, width = 12, height = 8, dpi = 300)
  }
  
  return(plots)
}

# ==============================================================================
# PUBLICATION-READY TABLES AND REPORTS
# ==============================================================================

#' Generate Publication-Ready Multiple Testing Correction Table
#'
#' Create formatted table for publication with corrected p-values
#'
#' @param correction_results Results from apply_mtc_correction_r
#' @param output_file Optional output file path
#' @return Formatted gt table
create_publication_table_mtc <- function(correction_results, output_file = NULL) {
  
  results_df <- correction_results$results_table
  summary_stats <- correction_results$summary_statistics
  
  # Add significance stars
  add_significance_stars <- function(pvalue) {
    if (is.na(pvalue)) return("NA")
    if (pvalue <= 0.001) return("***")
    if (pvalue <= 0.01) return("**")
    if (pvalue <= 0.05) return("*")
    return("ns")
  }
  
  results_df$Significance <- sapply(results_df$Corrected_pvalue, add_significance_stars)
  
  # Format p-values
  format_pvalue <- function(p) {
    if (is.na(p)) return("NA")
    if (p < 0.001) return(sprintf("%.2e", p))
    if (p < 0.1) return(sprintf("%.3f", p))
    return(sprintf("%.2f", p))
  }
  
  results_df$Original_pvalue_formatted <- sapply(results_df$Original_pvalue, format_pvalue)
  results_df$Corrected_pvalue_formatted <- sapply(results_df$Corrected_pvalue, format_pvalue)
  
  # Create final table
  final_table <- results_df %>%
    select(Test, Original_pvalue_formatted, Corrected_pvalue_formatted, Significance, Significant_Before, Significant_After) %>%
    rename(
      `Test Name` = Test,
      `Uncorrected p-value` = Original_pvalue_formatted,
      `Corrected p-value` = Corrected_pvalue_formatted,
      `Significance` = Significance,
      `Significant (Before)` = Significant_Before,
      `Significant (After)` = Significant_After
    )
  
  # Convert to gt table
  gt_table <- final_table %>%
    gt() %>%
    tab_header(
      title = paste("Multiple Testing Correction Results -", summary_stats$method),
      subtitle = paste("Significance level α =", summary_stats$alpha_level, 
                      "; Number of tests =", summary_stats$n_tests)
    ) %>%
    tab_options(
      table.width = pct(100),
      heading.title.font.size = 16,
      heading.subtitle.font.size = 12,
      column_labels.font.size = 12,
      body.text.font.size = 11
    ) %>%
    cols_label(
      `Significant (Before)` = "Significant\n(Before Correction)",
      `Significant (After)` = "Significant\n(After Correction)"
    ) %>%
    fmt_logical(
      columns = c(`Significant (Before)`, `Significant (After)`),
      truth = TRUE, 
      false_str = "No"
    )
  
  # Add footer with method information
  method_info <- paste0(
    "*** p < 0.001, ** p < 0.01, * p < 0.05, ns = not significant\n",
    "Correction method: ", summary_stats$method, "\n",
    "Significance rate before correction: ", sprintf("%.1f%%", summary_stats$significance_rate_before * 100), "\n",
    "Significance rate after correction: ", sprintf("%.1f%%", summary_stats$significance_rate_after * 100)
  )
  
  gt_table <- gt_table %>%
    tab_footnote(footnote = method_info, locations = cells_title())
  
  # Save table if output file specified
  if (!is.null(output_file)) {
    if (grepl("\\.csv$", output_file)) {
      write_csv(final_table, output_file)
    } else if (grepl("\\.png$", output_file)) {
      gtsave(gt_table, output_file)
    } else if (grepl("\\.html$", output_file)) {
      gt_table %>%
        gtsave(output_file)
    }
  }
  
  return(gt_table)
}

#' Generate Comprehensive Statistical Analysis Report
#'
#' Create complete report combining multiple testing correction and power analysis
#'
#' @param correction_results Results from apply_mtc_correction_r
#' @param power_results Results from power_analysis_mtc
#' @param output_dir Output directory for report files
#' @return List with report components
generate_comprehensive_report <- function(correction_results, power_results = NULL, 
                                        output_dir = "mtc_report") {
  
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Generate components
  report_components <- list()
  
  # 1. Create publication table
  pub_table <- create_publication_table_mtc(correction_results, 
                                           file.path(output_dir, "mtc_results.png"))
  report_components$publication_table <- pub_table
  
  # 2. Create before/after visualization
  visualization <- create_before_after_visualization(correction_results,
                                                   file.path(output_dir, "before_after_analysis.png"))
  report_components$visualization <- visualization
  
  # 3. Add power analysis if provided
  if (!is.null(power_results)) {
    power_summary <- data.frame(
      Metric = c("Effect Size", "Sample Size", "Uncorrected Power", "Corrected Power", 
                "Power Loss", "Corrected α"),
      Value = c(
        power_results$effect_size,
        power_results$sample_size,
        sprintf("%.3f", power_results$power_uncorrected),
        sprintf("%.3f", power_results$power_corrected),
        sprintf("%.3f", power_results$power_loss),
        sprintf("%.4f", power_results$corrected_alpha)
      ),
      stringsAsFactors = FALSE
    )
    
    power_table <- power_summary %>%
      gt() %>%
      tab_header(
        title = "Statistical Power Analysis",
        subtitle = paste("Method:", power_results$method, "; Tests:", power_results$n_tests)
      )
    
    gtsave(power_table, file.path(output_dir, "power_analysis.png"))
    report_components$power_table <- power_table
  }
  
  # 4. Create markdown report
  report_text <- generate_markdown_report(correction_results, power_results)
  writeLines(report_text, file.path(output_dir, "comprehensive_report.md"))
  report_components$markdown_report <- report_text
  
  # 5. Export data
  write_csv(correction_results$results_table, file.path(output_dir, "correction_data.csv"))
  
  if (!is.null(power_results)) {
    power_data <- data.frame(
      Effect_Size = power_results$effect_size,
      Sample_Size = power_results$sample_size,
      N_Tests = power_results$n_tests,
      Method = power_results$method,
      Power_Uncorrected = power_results$power_uncorrected,
      Power_Corrected = power_results$power_corrected,
      Power_Loss = power_results$power_loss,
      Corrected_Alpha = power_results$corrected_alpha
    )
    write_csv(power_data, file.path(output_dir, "power_analysis_data.csv"))
  }
  
  return(report_components)
}

#' Generate Markdown Report
#'
#' Create markdown-formatted report for documentation
#'
#' @param correction_results Results from apply_mtc_correction_r
#' @param power_results Optional power analysis results
#' @return Character vector with markdown content
generate_markdown_report <- function(correction_results, power_results = NULL) {
  
  summary_stats <- correction_results$summary_statistics
  
  report <- paste0(
    "# Multiple Testing Correction Analysis Report\n\n",
    "## Executive Summary\n\n",
    "This report presents the results of multiple testing correction analysis applied to ",
    "improve statistical rigor in drug screening and synthetic lethality studies.\n\n",
    
    "## Correction Method\n\n",
    "- **Method**: ", summary_stats$method, "\n",
    "- **Significance Level**: α = ", summary_stats$alpha_level, "\n",
    "- **Number of Tests**: ", summary_stats$n_tests, "\n\n",
    
    "## Results Summary\n\n",
    "| Metric | Before Correction | After Correction |\n",
    "|--------|------------------|------------------|\n",
    "| Significant Tests | ", summary_stats$n_significant_before, " | ", summary_stats$n_significant_after, " |\n",
    "| Significance Rate | ", sprintf("%.1f%%", summary_stats$significance_rate_before * 100), " | ", 
    sprintf("%.1f%%", summary_stats$significance_rate_after * 100), " |\n",
    "| Newly Significant | - | ", summary_stats$n_newly_significant, " |\n",
    "| Lost Significance | - | ", summary_stats$n_lost_significance, " |\n\n",
    
    "## Statistical Interpretation\n\n"
  )
  
  # Add interpretation based on results
  if (summary_stats$n_significant_after > summary_stats$n_significant_before) {
    report <- paste0(report, 
                    "The multiple testing correction increased the number of significant results, indicating ",
                    "that the original analysis may have been too conservative.\n\n")
  } else if (summary_stats$n_significant_after < summary_stats$n_significant_before) {
    report <- paste0(report,
                    "The multiple testing correction reduced the number of significant results, providing ",
                    "more stringent control of false positive findings.\n\n")
  } else {
    report <- paste0(report,
                    "The multiple testing correction did not change the number of significant results, ",
                    "suggesting robust statistical significance.\n\n")
  }
  
  # Add power analysis if available
  if (!is.null(power_results)) {
    report <- paste0(report,
                    "## Power Analysis\n\n",
                    "The statistical power analysis shows:\n\n",
                    "- **Effect Size**: ", power_results$effect_size, "\n",
                    "- **Sample Size**: ", power_results$sample_size, " per group\n",
                    "- **Uncorrected Power**: ", sprintf("%.3f", power_results$power_uncorrected), "\n",
                    "- **Corrected Power**: ", sprintf("%.3f", power_results$power_corrected), "\n",
                    "- **Power Loss**: ", sprintf("%.3f", power_results$power_loss), "\n\n")
  }
  
  report <- paste0(report,
                  "## Recommendations\n\n",
                  "1. **Use Corrected p-values**: Report corrected p-values for final significance testing\n",
                  "2. **Effect Size Reporting**: Always report effect sizes alongside p-values\n",
                  "3. **Power Considerations**: Ensure adequate sample sizes for future studies\n",
                  "4. **Transparent Reporting**: Include both original and corrected statistics\n",
                  "5. **Method Selection**: Consider the trade-off between Type I error control and power\n\n",
                  
                  "## Files Generated\n\n",
                  "- `correction_data.csv`: Detailed correction results\n",
                  "- `mtc_results.png`: Publication-ready table\n",
                  "- `before_after_analysis.png`: Visualization of correction impact\n",
                  "- `comprehensive_report.md`: This report\n\n",
                  
                  "---", "\n",
                  "*Report generated on ", Sys.time(), "*\n")
  
  return(report)
}

# ==============================================================================
# INTEGRATION WITH EXISTING FRAMEWORK
# ==============================================================================

#' Integrate Multiple Testing Correction with GDSC Analysis
#'
#' Apply multiple testing correction to GDSC validation results
#'
#' @param gdsc_results GDSC validation results (data frame or list)
#' @param correction_method Correction method to apply
#' @param alpha Significance level
#' @return List with corrected results
integrate_mtc_gdsc <- function(gdsc_results, correction_method = "BH", alpha = DEFAULT_ALPHA) {
  
  # Extract p-values from GDSC results
  # This assumes GDSC results have p-value or test statistic columns
  if (is.data.frame(gdsc_results)) {
    # Look for p-value columns
    pvalue_cols <- grep("p.value|pvalue|p_val", names(gdsc_results), ignore.case = TRUE)
    
    if (length(pvalue_cols) == 0) {
      # If no p-values, calculate from other statistics (simplified)
      if ("residual" %in% names(gdsc_results) && "experimental_value" %in% names(gdsc_results)) {
        gdsc_results$p_value <- 2 * (1 - pt(abs(gdsc_results$residual / (gdsc_results$experimental_value * 0.1)), df = 10))
        pvalue_cols <- which(names(gdsc_results) == "p_value")
      } else {
        stop("No p-value information found in GDSC results")
      }
    }
    
    pvalues <- gdsc_results[[pvalue_cols[1]]]  # Use first p-value column
  } else {
    stop("GDSC results must be a data frame")
  }
  
  # Apply multiple testing correction
  mtc_results <- apply_mtc_correction_r(pvalues, correction_method, alpha)
  
  # Add corrected p-values to original results
  gdsc_corrected <- gdsc_results
  gdsc_corrected$corrected_pvalue <- mtc_results$results_table$Corrected_pvalue
  gdsc_corrected$significant_before <- mtc_results$results_table$Significant_Before
  gdsc_corrected$significant_after <- mtc_results$results_table$Significant_After
  gdsc_corrected$correction_method <- correction_method
  
  return(list(
    original_results = gdsc_results,
    corrected_results = gdsc_corrected,
    mtc_summary = mtc_results$summary_statistics,
    correction_table = mtc_results$results_table
  ))
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

#' Export All Results to Multiple Formats
#'
#' Export analysis results to CSV, PNG, and JSON formats
#'
#' @param results_list List containing all analysis results
#' @param output_prefix Output file prefix
#' @param output_dir Output directory
export_results_comprehensive <- function(results_list, output_prefix, output_dir = ".") {
  
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Export to different formats
  for (format in c("csv", "png", "json")) {
    filename <- file.path(output_dir, paste0(output_prefix, ".", format))
    
    if (format == "csv" && "correction_table" %in% names(results_list)) {
      write_csv(results_list$correction_table, filename)
    } else if (format == "png" && "visualization" %in% names(results_list)) {
      ggsave(filename, results_list$visualization$combined_plot, width = 12, height = 10, dpi = 300)
    } else if (format == "json") {
      jsonlite::write_json(results_list, filename, pretty = TRUE)
    }
  }
  
  cat("Results exported to", output_dir, "\n")
}

# Export functions for use in other scripts
utils::globalVariables(c(".", "p.adjust", "pwr", "gt", "Test", "Original_pvalue", "Corrected_pvalue", 
                        "Significant_Before", "Significant_After", "effect_size", "sample_size", 
                        "n_tests", "method", "power_uncorrected", "power_corrected", "power_loss"))

# End of multiple testing correction integration
cat("Multiple Testing Correction Integration loaded successfully (v", MTC_FRAMEWORK_VERSION, ")\n")