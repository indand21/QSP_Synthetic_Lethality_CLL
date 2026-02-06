# Publication-Quality Visualization and Output Formatting
# ==============================================================================
# This R script provides comprehensive visualization capabilities
# for creating publication-quality figures for synthetic lethality research
#
# Features:
# - Advanced ggplot2-based visualizations
# - Publication-ready figure formatting
# - Interactive plots for exploratory analysis
# - Automated figure generation workflows
# - Standardized color schemes and themes
# ==============================================================================

# Load required packages
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(viridis)
  library(RColorBrewer)
  library(gridExtra)
  library(grid)
  library(gtable)
  library(ggpubr)
  library(ComplexHeatmap)
  library(plotly)
  library(DT)
  library(shiny)
  library(ggplotify)
  library(cowplot)
  library(patchwork)
  library(scales)
  library(extrafont)
  library(ggrepel)
  library(ggfortify)
  library(GGally)
})

# ==============================================================================
# PUBLICATION THEME SYSTEM
# ==============================================================================

#' Define publication-quality themes for different figure types
#'
#' @param base_size Base font size
#' @param base_family Font family
#' @param aspect_ratio Aspect ratio for plots
#' @return ggplot theme object

# Main publication theme
theme_publication <- function(base_size = 12, base_family = "Arial", aspect_ratio = 16/9) {
  theme_minimal(base_size = base_size, base_family = base_family) %+replace%
    theme(
      # Text formatting
      plot.title = element_text(face = "bold", size = rel(1.2), hjust = 0, family = base_family),
      plot.subtitle = element_text(size = rel(1.0), hjust = 0, family = base_family),
      plot.caption = element_text(size = rel(0.8), hjust = 0, family = base_family),
      axis.title = element_text(face = "bold", size = rel(1.0), family = base_family),
      axis.text = element_text(size = rel(0.9), family = base_family),
      axis.text.x = element_text(margin = margin(t = 5, b = 5)),
      axis.text.y = element_text(margin = margin(r = 5, l = 5)),
      
      # Legend formatting
      legend.title = element_text(face = "bold", size = rel(0.9), family = base_family),
      legend.text = element_text(size = rel(0.8), family = base_family),
      legend.margin = margin(0, 0, 0, 0),
      legend.key.size = unit(0.8, "lines"),
      legend.spacing = unit(0.5, "lines"),
      
      # Panel and grid formatting
      panel.grid.major = element_line(color = "gray90", size = 0.3),
      panel.grid.minor = element_line(color = "gray95", size = 0.2),
      panel.border = element_rect(color = "black", fill = NA, size = 0.5),
      panel.background = element_rect(color = "white", fill = "white"),
      
      # Strip formatting
      strip.text = element_text(face = "bold", size = rel(0.9), family = base_family),
      strip.background = element_rect(fill = "gray90", color = "gray80"),
      
      # Plot margins and aspect ratio
      plot.margin = margin(10, 10, 10, 10),
      aspect.ratio = aspect_ratio
    )
}

# Theme for supplementary figures (smaller size)
theme_supplementary <- function(base_size = 10, base_family = "Arial", aspect_ratio = 4/3) {
  theme_publication(base_size = base_size, base_family = base_family, aspect_ratio = aspect_ratio) %+replace%
    theme(
      plot.title = element_text(face = "bold", size = rel(1.1)),
      axis.text = element_text(size = rel(0.8)),
      legend.text = element_text(size = rel(0.7))
    )
}

# Theme for main figures (larger size)
theme_main_figure <- function(base_size = 14, base_family = "Arial", aspect_ratio = 4/3) {
  theme_publication(base_size = base_size, base_family = base_family, aspect_ratio = aspect_ratio) %+replace%
    theme(
      plot.title = element_text(face = "bold", size = rel(1.3)),
      axis.title = element_text(face = "bold", size = rel(1.1)),
      legend.title = element_text(face = "bold", size = rel(1.0)),
      legend.text = element_text(size = rel(0.9))
    )
}

# ==============================================================================
# COLOR SCHEMES AND PALETTES
# ==============================================================================

#' Define standardized color palettes for different data types
#'
#' @return List of color palettes

get_color_palettes <- function() {
  palettes <- list(
    # Synthetic lethality scores
    sl_scores = colorRampPalette(c("#440154", "#31688e", "#35b779", "#fde725"))(100),
    
    # Treatment conditions
    treatment_conditions = c(
      "Control" = "#666666",
      "Low_Dose" = "#1f78b4", 
      "Medium_Dose" = "#33a02c",
      "High_Dose" = "#e31a1c"
    ),
    
    # Cell types
    cell_types = c(
      "WT" = "#1f77b4",
      "ATM_proficient" = "#1f77b4",
      "ATM_deficient" = "#d62728",
      "ATM_def" = "#d62728"
    ),
    
    # Target pathways
    target_pathways = c(
      "PARP" = "#e31a1c",
      "ATR" = "#ff7f00", 
      "CHK1" = "#ffff33",
      "WEE1" = "#1f78b4",
      "ATM" = "#6a3d9a",
      "DNA_PK" = "#b15928"
    ),
    
    # Pathway activity levels
    pathway_activity = c(
      "Low" = "#440154",
      "Medium" = "#31688e", 
      "High" = "#35b779"
    ),
    
    # Validation metrics
    validation_metrics = c(
      "Excellent" = "#2ca02c",
      "Good" = "#98df8a",
      "Fair" = "#ffbb78", 
      "Poor" = "#ff7f0e",
      "Failed" = "#d62728"
    )
  )
  
  return(palettes)
}

#' Create custom color scale for synthetic lethality scores
#'
#' @param palette Color palette
#' @return ggplot color scale

scale_color_sl_score <- function(palette = "sl_scores") {
  palettes <- get_color_palettes()
  scale_color_gradientn(colors = palettes[[palette]], name = "SL Score")
}

scale_fill_sl_score <- function(palette = "sl_scores") {
  palettes <- get_color_palettes()
  scale_fill_gradientn(colors = palettes[[palette]], name = "SL Score")
}

# ==============================================================================
# CORE VISUALIZATION FUNCTIONS
# ==============================================================================

#' Create comprehensive synthetic lethality score visualization
#'
#' @param screening_results Data frame with screening results
#' @param figure_type Type of figure ("main", "supplementary", "interactive")
#' @param show_targets Whether to color by target pathway
#' @return ggplot object
create_sl_score_visualization <- function(screening_results, 
                                         figure_type = "main",
                                         show_targets = TRUE) {
  
  # Select appropriate theme
  if (figure_type == "main") {
    theme_func <- theme_main_figure
  } else if (figure_type == "supplementary") {
    theme_func <- theme_supplementary  
  } else {
    theme_func <- theme_publication
  }
  
  # Prepare data
  plot_data <- screening_results %>%
    arrange(desc(Synthetic_Lethality_Score)) %>%
    mutate(
      rank = row_number(),
      target_clean = str_replace_all(Target, "_", " "),
      sl_category = case_when(
        Synthetic_Lethality_Score > 2.0 ~ "High",
        Synthetic_Lethality_Score > 1.5 ~ "Medium", 
        TRUE ~ "Low"
      )
    )
  
  # Create main plot
  if (show_targets && "Target" %in% names(screening_results)) {
    p <- ggplot(plot_data, aes(x = rank, y = Synthetic_Lethality_Score, 
                              color = Target, size = Therapeutic_Index)) +
      geom_point(alpha = 0.7) +
      scale_color_manual(values = get_color_palettes()$target_pathways, 
                        name = "Target Pathway") +
      scale_size_continuous(name = "Therapeutic\nIndex", range = c(1, 4))
  } else {
    p <- ggplot(plot_data, aes(x = rank, y = Synthetic_Lethality_Score, 
                              color = sl_category)) +
      geom_point(alpha = 0.7, size = 2) +
      scale_color_manual(values = get_color_palettes()$pathway_activity,
                        name = "SL Score\nCategory") +
      scale_size_continuous(range = c(1, 4))
  }
  
  # Add reference lines
  p <- p +
    geom_hline(yintercept = 1, linetype = "dashed", color = "gray50", size = 1) +
    geom_hline(yintercept = 2, linetype = "dotted", color = "orange", size = 1) +
    
    # Formatting
    labs(
      title = "Drug Ranking by Synthetic Lethality Score",
      subtitle = "ATM-deficient CLL vs. ATM-proficient cells",
      x = "Drug Rank",
      y = "Synthetic Lethality Score",
      caption = "Dashed line: SL = 1 (no selectivity); Dotted line: SL = 2 (high selectivity)"
    ) +
    theme_func() +
    theme(
      legend.position = ifelse(show_targets, "right", "bottom"),
      panel.grid.minor = element_blank()
    )
  
  return(p)
}

#' Create pathway activity heatmap
#'
#' @param pathway_data Data frame with pathway activity data
#' @param figure_type Type of figure
#' @return Heatmap object
create_pathway_heatmap <- function(pathway_data, figure_type = "main") {
  
  # Prepare data for heatmap
  heatmap_data <- pathway_data %>%
    select(Drug, Target, DSB_Level, HR_Activity, PARP_Activity, 
           Cell_Cycle_Arrest, Synthetic_Lethality_Score) %>%
    filter(complete.cases(.)) %>%
    column_to_rownames("Drug")
  
  # Create color palette
  color_pal <- colorRampPalette(c("#440154", "#31688e", "#35b779", "#fde725"))(100)
  
  # Create heatmap
  hm <- Heatmap(
    as.matrix(heatmap_data),
    name = "Activity Level",
    col = color_pal,
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    show_row_names = TRUE,
    show_column_names = TRUE,
    row_names_gp = gpar(fontsize = ifelse(figure_type == "main", 10, 8)),
    column_names_gp = gpar(fontsize = ifelse(figure_type == "main", 11, 9)),
    heatmap_legend_param = list(
      title = "Normalized Activity",
      title_gp = gpar(fontsize = ifelse(figure_type == "main", 12, 10)),
      labels_gp = gpar(fontsize = ifelse(figure_type == "main", 10, 8))
    )
  )
  
  return(hm)
}

#' Create model validation visualization
#'
#' @param validation_results Model validation results
#' @param figure_type Type of figure
#' @return Combined validation plot
create_model_validation_plot <- function(validation_results, figure_type = "main") {
  
  # Select appropriate theme
  if (figure_type == "main") {
    theme_func <- theme_main_figure
  } else if (figure_type == "supplementary") {
    theme_func <- theme_supplementary
  } else {
    theme_func <- theme_publication
  }
  
  validation_data <- validation_results$validation_data
  
  # Create multi-panel validation plot
  p1 <- ggplot(validation_data, aes(x = experimental, y = simulated)) +
    geom_point(alpha = 0.6, color = "steelblue", size = 2) +
    geom_smooth(method = "lm", se = TRUE, color = "red") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
    labs(
      title = "A. Model Prediction vs. Experimental Data",
      x = "Experimental Values",
      y = "Predicted Values",
      subtitle = paste("R² =", round(validation_results$basic_metrics$r_squared, 3))
    ) +
    theme_func(aspect_ratio = 1)
  
  # Bland-Altman plot
  validation_data$mean_values <- (validation_data$experimental + validation_data$simulated) / 2
  validation_data$difference <- validation_data$experimental - validation_data$simulated
  
  p2 <- ggplot(validation_data, aes(x = mean_values, y = difference)) +
    geom_point(alpha = 0.6, color = "steelblue", size = 2) +
    geom_hline(yintercept = 0, linetype = "solid", color = "black") +
    geom_hline(yintercept = validation_results$bland_altman$mean_difference, 
               linetype = "solid", color = "red") +
    geom_hline(yintercept = validation_results$bland_altman$loa_upper, 
               linetype = "dashed", color = "red") +
    geom_hline(yintercept = validation_results$bland_altman$loa_lower, 
               linetype = "dashed", color = "red") +
    labs(
      title = "B. Bland-Altman Analysis",
      x = "Mean of Experimental and Predicted",
      y = "Difference (Experimental - Predicted)"
    ) +
    theme_func(aspect_ratio = 1) +
    theme(legend.position = "none")
  
  # Combine plots
  combined_plot <- p1 + p2 + plot_layout(ncol = 2)
  
  return(combined_plot)
}

#' Create time-course visualization
#'
#' @param timecourse_data Data frame with time-course data
#' @param variables Variables to plot
#' @param figure_type Type of figure
#' @return Time-course plot
create_timecourse_visualization <- function(timecourse_data, 
                                          variables = c("ApoptosisSignal", "DSB", "RAD51_focus"),
                                          figure_type = "main") {
  
  # Select appropriate theme
  if (figure_type == "main") {
    theme_func <- theme_main_figure
  } else if (figure_type == "supplementary") {
    theme_func <- theme_supplementary
  } else {
    theme_func <- theme_publication
  }
  
  # Prepare data for plotting
  plot_data <- timecourse_data %>%
    select(Time, all_of(variables)) %>%
    pivot_longer(cols = all_of(variables), names_to = "variable", values_to = "value")
  
  # Create plot
  p <- ggplot(plot_data, aes(x = Time, y = value, color = variable)) +
    geom_line(size = 1, alpha = 0.8) +
    geom_point(size = 1, alpha = 0.6) +
    scale_color_viridis_d(name = "Model State", option = "plasma") +
    labs(
      title = "QSP Model Time-Course Simulation",
      subtitle = "DDR pathway dynamics in ATM-deficient CLL",
      x = "Time (hours)",
      y = "Normalized Concentration"
    ) +
    theme_func() +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )
  
  return(p)
}

#' Create dose-response curve visualization
#'
#' @param dose_response_data Data frame with dose-response data
#' @param drug_name Drug to plot (optional)
#' @param figure_type Type of figure
#' @return Dose-response plot
create_dose_response_plot <- function(dose_response_data, 
                                     drug_name = NULL,
                                     figure_type = "main") {
  
  # Select appropriate theme
  if (figure_type == "main") {
    theme_func <- theme_main_figure
  } else if (figure_type == "supplementary") {
    theme_func <- theme_supplementary
  } else {
    theme_func <- theme_publication
  }
  
  # Filter data if drug specified
  if (!is.null(drug_name)) {
    plot_data <- dose_response_data %>% filter(Drug == drug_name)
  } else {
    plot_data <- dose_response_data
  }
  
  # Create plot
  p <- ggplot(plot_data, aes(x = log10(Dose), y = Response, 
                            color = Cell_Type, shape = Drug)) +
    geom_point(size = 2, alpha = 0.7) +
    geom_smooth(method = "nls", se = FALSE, 
                formula = y ~ Bottom + (Top - Bottom)/(1 + 10^((LogIC50 - x)*HillSlope)),
                color = "black") +
    scale_color_manual(values = get_color_palettes()$cell_types, name = "Cell Type") +
    labs(
      title = ifelse(is.null(drug_name), 
                    "Dose-Response Analysis", 
                    paste("Dose-Response:", drug_name)),
      x = "log₁₀(Dose)",
      y = "Response (% apoptosis)"
    ) +
    theme_func() +
    theme(
      panel.grid.minor = element_blank(),
      legend.position = "right"
    )
  
  return(p)
}

# ==============================================================================
# INTERACTIVE VISUALIZATIONS
# ==============================================================================

#' Create interactive synthetic lethality score plot
#'
#' @param screening_results Data frame with screening results
#' @return plotly object
create_interactive_sl_plot <- function(screening_results) {
  
  # Prepare data
  plot_data <- screening_results %>%
    mutate(
      hover_text = paste0(
        "Drug: ", Drug, "\n",
        "Target: ", Target, "\n",
        "SL Score: ", round(Synthetic_Lethality_Score, 2), "\n",
        "Therapeutic Index: ", round(Therapeutic_Index, 2)
      )
    )
  
  # Create interactive plot
  p <- plot_ly(
    data = plot_data,
    x = ~Synthetic_Lethality_Score,
    y = ~Therapeutic_Index,
    color = ~Target,
    size = ~Apoptosis_ATM_def,
    text = ~hover_text,
    hovertemplate = "%{text}<extra></extra>",
    type = "scatter",
    mode = "markers",
    marker = list(
      sizemode = "diameter",
      line = list(width = 1, color = "white")
    )
  ) %>%
    layout(
      title = "Interactive Drug Screening Results",
      xaxis = list(title = "Synthetic Lethality Score"),
      yaxis = list(title = "Therapeutic Index"),
      hovermode = "closest"
    )
  
  return(p)
}

#' Create interactive validation plot
#'
#' @param validation_results Model validation results
#' @return plotly object
create_interactive_validation_plot <- function(validation_results) {
  
  validation_data <- validation_results$validation_data
  
  # Create subplots
  fig <- subplot(
    # Scatter plot
    plot_ly(
      data = validation_data,
      x = ~experimental,
      y = ~simulated,
      type = "scatter",
      mode = "markers",
      name = "Data Points",
      hovertemplate = "Experimental: %{x:.2f}<br>Predicted: %{y:.2f}<extra></extra>"
    ) %>%
      add_lines(
        x = range(validation_data$experimental),
        y = range(validation_data$experimental),
        line = list(color = "red", dash = "dash"),
        name = "Perfect Agreement"
      ),
    
    # Bland-Altman plot  
    plot_ly(
      data = validation_data,
      x = ~(experimental + simulated) / 2,
      y = ~experimental - simulated,
      type = "scatter",
      mode = "markers",
      name = "Differences",
      hovertemplate = "Mean: %{x:.2f}<br>Difference: %{y:.2f}<extra></extra>"
    ) %>%
      add_lines(
        x = range((validation_data$experimental + validation_data$simulated) / 2),
        y = c(0, 0),
        line = list(color = "red"),
        name = "Zero Difference"
      ),
    
    nrows = 1,
    shareX = FALSE,
    shareY = FALSE,
    titleX = TRUE,
    titleY = TRUE
  ) %>%
    layout(
      title = "Interactive Model Validation",
      showlegend = FALSE
    )
  
  return(fig)
}

# ==============================================================================
# FIGURE COMPOSITION AND EXPORT
# ==============================================================================

#' Create multi-panel figure composition
#'
#' @param plot_list List of plots to combine
#' @param layout Layout specification
#' @param figure_type Type of figure
#' @return Composed figure
compose_figure <- function(plot_list, 
                          layout = "grid", 
                          figure_type = "main") {
  
  if (layout == "grid") {
    # Grid layout
    if (length(plot_list) <= 2) {
      composed <- plot_list[[1]] + plot_list[[2]] + plot_layout(ncol = 2)
    } else if (length(plot_list) <= 4) {
      composed <- wrap_plots(plot_list, ncol = 2, nrow = 2)
    } else {
      composed <- wrap_plots(plot_list, ncol = 3, nrow = 2)
    }
  } else if (layout == "custom") {
    # Custom layout - would need specific layout specification
    composed <- plot_list[[1]]
  }
  
  return(composed)
}

#' Export figure with multiple formats and resolutions
#'
#' @param plot ggplot object or list of plots
#' @param filename Base filename (without extension)
#' @param output_dir Output directory
#' @param formats List of formats to export ("png", "pdf", "svg", "tiff")
#' @param dpi Resolution for raster formats
#' @param width Figure width in inches
#' @param height Figure height in inches
#' @export
export_figure <- function(plot, 
                         filename, 
                         output_dir = "figures",
                         formats = c("png", "pdf"),
                         dpi = 300,
                         width = 8,
                         height = 6) {
  
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Export each format
  for (format in formats) {
    full_path <- file.path(output_dir, paste0(filename, ".", format))
    
    if (format == "png") {
      ggsave(full_path, plot, width = width, height = height, dpi = dpi, bg = "white")
    } else if (format == "pdf") {
      ggsave(full_path, plot, width = width, height = height, bg = "white")
    } else if (format == "svg") {
      ggsave(full_path, plot, width = width, height = height, bg = "white")
    } else if (format == "tiff") {
      ggsave(full_path, plot, width = width, height = height, dpi = dpi, bg = "white")
    }
  }
  
  cat("Figure exported to:", output_dir, "\n")
  cat("Formats:", paste(formats, collapse = ", "), "\n")
}

#' Create figure panels for manuscript
#'
#' @param screening_results Drug screening results
#' @param validation_results Model validation results
#' @param timecourse_data Time-course simulation data
#' @param output_dir Output directory
#' @return List of figure file paths
create_manuscript_figures <- function(screening_results,
                                    validation_results = NULL,
                                    timecourse_data = NULL,
                                    output_dir = "manuscript_figures") {
  
  figure_files <- list()
  
  cat("Creating manuscript figures...\n")
  
  # Figure 1: Model overview and validation
  if (!is.null(validation_results)) {
    fig1_plots <- list(
      create_model_validation_plot(validation_results, "main")
    )
    
    fig1 <- compose_figure(fig1_plots, "grid", "main")
    export_figure(fig1, "Figure1_Model_Validation", output_dir, 
                 formats = c("png", "pdf"), width = 12, height = 6)
    figure_files$Figure1 <- file.path(output_dir, "Figure1_Model_Validation.png")
  }
  
  # Figure 2: Drug screening results
  fig2_plots <- list(
    create_sl_score_visualization(screening_results, "main", TRUE)
  )
  
  fig2 <- compose_figure(fig2_plots, "grid", "main")
  export_figure(fig2, "Figure2_Drug_Screening", output_dir, 
               formats = c("png", "pdf"), width = 10, height = 8)
  figure_files$Figure2 <- file.path(output_dir, "Figure2_Drug_Screening.png")
  
  # Figure 3: Pathway analysis
  if (!is.null(timecourse_data)) {
    fig3_plots <- list(
      create_timecourse_visualization(timecourse_data, c("ApoptosisSignal", "DSB", "RAD51_focus"), "main")
    )
    
    fig3 <- compose_figure(fig3_plots, "grid", "main")
    export_figure(fig3, "Figure3_Pathway_Dynamics", output_dir, 
                 formats = c("png", "pdf"), width = 10, height = 6)
    figure_files$Figure3 <- file.path(output_dir, "Figure3_Pathway_Dynamics.png")
  }
  
  # Supplementary Figures
  cat("Creating supplementary figures...\n")
  
  # Supplementary Figure 1: Detailed validation analysis
  if (!is.null(validation_results)) {
    supp1_plots <- list(
      create_model_validation_plot(validation_results, "supplementary")
    )
    
    supp1 <- compose_figure(supp1_plots, "grid", "supplementary")
    export_figure(supp1, "Supplementary_Figure1_Validation_Details", 
                 output_dir, formats = c("png", "pdf"), width = 12, height = 6)
    figure_files$Supplementary1 <- file.path(output_dir, "Supplementary_Figure1_Validation_Details.png")
  }
  
  # Supplementary Figure 2: Drug screening by target
  supp2_plots <- list(
    create_sl_score_visualization(screening_results, "supplementary", FALSE)
  )
  
  supp2 <- compose_figure(supp2_plots, "grid", "supplementary")
  export_figure(supp2, "Supplementary_Figure2_Drug_Analysis", 
               output_dir, formats = c("png", "pdf"), width = 10, height = 8)
  figure_files$Supplementary2 <- file.path(output_dir, "Supplementary_Figure2_Drug_Analysis.png")
  
  cat("Manuscript figures created successfully!\n")
  return(figure_files)
}

# ==============================================================================
# STATISTICAL SUMMARIES AND TABLES
# ==============================================================================

#' Create publication-ready statistical summary table
#'
#' @param analysis_results Analysis results object
#' @param table_type Type of table ("main", "supplementary")
#' @return Formatted table
create_statistical_summary_table <- function(analysis_results, table_type = "main") {
  
  if (is.null(analysis_results$sl_analysis)) {
    stop("No synthetic lethality analysis results found")
  }
  
  sl_results <- analysis_results$sl_analysis
  
  # Main results table
  if (table_type == "main") {
    table_data <- sl_results$descriptive_statistics %>%
      mutate(
        `Mean SL Score` = paste0(round(mean_sl_score, 2), " ± ", round(sd_sl_score, 2)),
        `Median (IQR)` = paste0(round(median_sl_score, 2), " (", 
                               round(q25_sl_score, 2), "-", round(q75_sl_score, 2), ")"),
        `High SL (n)` = paste0(high_synthetic_lethality, " (", 
                              round(prop_high_sl * 100, 1), "%)")
      ) %>%
      select(`Mean SL Score`, `Median (IQR)`, `High SL (n)`) %>%
      rownames_to_column("Target")
    
  } else {
    # Detailed results table
    table_data <- sl_results$descriptive_statistics %>%
      mutate(
        `Mean ± SD` = paste0(round(mean_sl_score, 3), " ± ", round(sd_sl_score, 3)),
        `Median (IQR)` = paste0(round(median_sl_score, 3), " (", 
                               round(q25_sl_score, 3), "-", round(q75_sl_score, 3), ")"),
        `Range` = paste0(round(min_sl_score, 3), " - ", round(max_sl_score, 3)),
        `High SL (n, %)` = paste0(high_synthetic_lethality, " (", 
                                 round(prop_high_sl * 100, 1), "%)")
      ) %>%
      select(`Mean ± SD`, `Median (IQR)`, `Range`, `High SL (n, %)`) %>%
      rownames_to_column("Target")
  }
  
  # Format as gt table
  table_formatted <- table_data %>%
    gt(rowname_col = "Target") %>%
    tab_header(
      title = ifelse(table_type == "main", 
                    "Synthetic Lethality Analysis by Target Pathway",
                    "Detailed Statistical Summary by Target Pathway"),
      subtitle = "ATM-deficient vs. ATM-proficient CLL cells"
    ) %>%
    tab_options(
      table.width = pct(100),
      heading.title.font.size = 16,
      heading.subtitle.font.size = 12,
      column_labels.font.size = 12,
      body.text.font.size = 11
    ) %>%
    cols_label(
      `Mean SL Score` = "Mean ± SD",
      `Median (IQR)` = "Median (IQR)",
      `High SL (n)` = "High SL (n, %)"
    )
  
  return(table_formatted)
}

#' Create validation metrics table
#'
#' @param validation_results Model validation results
#' @return Formatted validation table
create_validation_table <- function(validation_results) {
  
  # Extract key metrics
  metrics <- data.frame(
    Metric = c("RMSE", "MAE", "R²", "Correlation", "p-value", "Mean Difference", "95% LoA"),
    Value = c(
      round(validation_results$basic_metrics$rmse, 3),
      round(validation_results$basic_metrics$mae, 3),
      round(validation_results$basic_metrics$r_squared, 3),
      round(validation_results$basic_metrics$correlation, 3),
      round(validation_results$basic_metrics$correlation_p_value, 4),
      round(validation_results$bland_altman$mean_difference, 3),
      paste0("[", round(validation_results$bland_altman$loa_lower, 3), ", ", 
             round(validation_results$bland_altman$loa_upper, 3), "]")
    ),
    Interpretation = c(
      "Lower is better (< 2.0 good)",
      "Lower is better (< 1.5 good)", 
      "Higher is better (> 0.8 good)",
      "Higher is better (> 0.8 strong)",
      "p < 0.05 significant",
      "Near zero indicates good agreement",
      "95% of points within range"
    )
  )
  
  # Format as gt table
  table_formatted <- metrics %>%
    gt() %>%
    tab_header(
      title = "QSP Model Validation Metrics",
      subtitle = "Comparison with experimental data"
    ) %>%
    tab_options(
      table.width = pct(100),
      heading.title.font.size = 16,
      column_labels.font.size = 12,
      body.text.font.size = 11
    ) %>%
    cols_label(
      Metric = "Validation Metric",
      Value = "Value",
      Interpretation = "Interpretation"
    )
  
  return(table_formatted)
}

# ==============================================================================
# AUTOMATED FIGURE GENERATION WORKFLOW
# ==============================================================================

#' Complete figure generation workflow
#'
#' @param screening_results Drug screening results
#' @param validation_results Model validation results
#' @param timecourse_data Time-course simulation data
#' @param output_dir Output directory
#' @param figure_types Types of figures to generate
#' @return List of generated figure files
complete_figure_generation <- function(screening_results,
                                     validation_results = NULL,
                                     timecourse_data = NULL,
                                     output_dir = "all_figures",
                                     figure_types = c("manuscript", "interactive", "supplementary")) {
  
  cat("Starting complete figure generation workflow...\n")
  
  # Create output directories
  dirs_to_create <- c(output_dir)
  if ("interactive" %in% figure_types) {
    dirs_to_create <- c(dirs_to_create, file.path(output_dir, "interactive"))
  }
  if ("manuscript" %in% figure_types) {
    dirs_to_create <- c(dirs_to_create, file.path(output_dir, "manuscript"))
  }
  
  for (dir in dirs_to_create) {
    if (!dir.exists(dir)) {
      dir.create(dir, recursive = TRUE)
    }
  }
  
  generated_files <- list()
  
  # Generate manuscript figures
  if ("manuscript" %in% figure_types) {
    cat("Generating manuscript figures...\n")
    generated_files$manuscript <- create_manuscript_figures(
      screening_results, validation_results, timecourse_data,
      file.path(output_dir, "manuscript")
    )
  }
  
  # Generate interactive figures
  if ("interactive" %in% figure_types) {
    cat("Generating interactive figures...\n")
    
    # Interactive SL plot
    interactive_sl <- create_interactive_sl_plot(screening_results)
    htmlwidgets::saveWidget(interactive_sl, 
                          file.path(output_dir, "interactive", "sl_interactive.html"))
    generated_files$interactive$sl_plot <- file.path(output_dir, "interactive", "sl_interactive.html")
    
    # Interactive validation plot
    if (!is.null(validation_results)) {
      interactive_val <- create_interactive_validation_plot(validation_results)
      htmlwidgets::saveWidget(interactive_val,
                            file.path(output_dir, "interactive", "validation_interactive.html"))
      generated_files$interactive$validation_plot <- file.path(output_dir, "interactive", "validation_interactive.html")
    }
  }
  
  # Generate supplementary figures
  if ("supplementary" %in% figure_types) {
    cat("Generating supplementary figures...\n")
    
    # Additional analysis plots
    supp_plots <- list(
      create_sl_score_visualization(screening_results, "supplementary", FALSE),
      create_dose_response_plot(screening_results, figure_type = "supplementary")
    )
    
    for (i in seq_along(supp_plots)) {
      export_figure(supp_plots[[i]], 
                   paste0("Supplementary_Figure_", i),
                   file.path(output_dir, "manuscript"),
                   formats = c("png", "pdf"),
                   width = 8, height = 6)
    }
  }
  
  # Generate statistical tables
  cat("Generating statistical tables...\n")
  if (!is.null(validation_results)) {
    validation_table <- create_validation_table(validation_results)
    gtsave(validation_table, 
           file.path(output_dir, "manuscript", "validation_table.png"))
  }
  
  # Create summary report
  cat("Creating figure generation summary...\n")
  summary_report <- paste0(
    "Figure Generation Summary\n",
    "========================\n\n",
    "Generated at: ", Sys.time(), "\n",
    "Output directory: ", output_dir, "\n\n",
    "Manuscript figures: ", length(generated_files$manuscript %||% list()), "\n",
    "Interactive figures: ", length(generated_files$interactive %||% list()), "\n\n",
    "Files generated:\n",
    paste(names(unlist(generated_files)), collapse = "\n")
  )
  
  writeLines(summary_report, file.path(output_dir, "figure_generation_summary.txt"))
  
  cat("Complete figure generation workflow finished!\n")
  cat("Output directory:", output_dir, "\n")
  
  return(generated_files)
}

# Export functions for use in other scripts
utils::globalVariables(c(".", "Time", "ApoptosisSignal", "DSB", "RAD51_focus", "Drug", 
                       "Synthetic_Lethality_Score", "Target", "rank", "target_clean", 
                       "sl_category", "experimental", "simulated", "Therapeutic_Index", 
                       "Apoptosis_WT", "Apoptosis_ATM_def", "variable", "value", 
                       "Response", "Cell_Type", "Dose", "LogIC50", "HillSlope"))

# End of publication-quality visualization script
# ==============================================================================
# ENHANCED STATISTICAL INTEGRATION METHODS
# ==============================================================================

#' Create Statistical Integration Visualization
#'
#' Generate comprehensive visualizations for integrated statistical analysis
#'
#' @param integrated_results Results from integrated statistical pipeline
#' @param output_dir Output directory for plots
#' @return List of generated plot objects
create_statistical_integration_visualization <- function(integrated_results, 
                                                        output_dir = "statistical_plots") {
  
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  plots <- list()
  
  # 1. Multiple testing correction visualization
  if ("enhanced_synthetic_lethality" %in% names(integrated_results)) {
    sl_data <- integrated_results$enhanced_synthetic_lethality
    if ("multiple_testing" %in% names(sl_data)) {
      mtc_data <- sl_data$multiple_testing
      
      # Create correction comparison plot
      correction_plot <- create_mtc_comparison_plot(mtc_data)
      plots$mtc_comparison <- correction_plot
      
      # Save plot
      ggsave(file.path(output_dir, "multiple_testing_correction.png"),
             correction_plot, width = 12, height = 8, dpi = 300)
    }
  }
  
  # 2. Effect size visualization
  if ("enhanced_synthetic_lethality" %in% names(integrated_results)) {
    sl_data <- integrated_results$enhanced_synthetic_lethality
    if ("effect_sizes" %in% names(sl_data)) {
      effect_data <- sl_data$effect_sizes
      
      effect_plot <- create_effect_size_visualization(effect_data)
      plots$effect_sizes <- effect_plot
      
      ggsave(file.path(output_dir, "effect_size_analysis.png"),
             effect_plot, width = 12, height = 8, dpi = 300)
    }
  }
  
  # 3. Model performance assessment
  if ("enhanced_validation" %in% names(integrated_results)) {
    val_data <- integrated_results$enhanced_validation
    if ("performance_assessment" %in% names(val_data)) {
      perf_data <- val_data$performance_assessment
      
      performance_plot <- create_performance_assessment_plot(perf_data)
      plots$performance <- performance_plot
      
      ggsave(file.path(output_dir, "model_performance_assessment.png"),
             performance_plot, width = 12, height = 8, dpi = 300)
    }
  }
  
  # 4. Statistical rigor assessment
  if ("enhanced_validation" %in% names(integrated_results)) {
    val_data <- integrated_results$enhanced_validation
    if ("significance_tests" %in% names(val_data)) {
      rigor_plot <- create_statistical_rigor_plot(val_data)
      plots$rigor <- rigor_plot
      
      ggsave(file.path(output_dir, "statistical_rigor_assessment.png"),
             rigor_plot, width = 12, height = 8, dpi = 300)
    }
  }
  
  # 5. Bootstrap confidence intervals
  if ("enhanced_validation" %in% names(integrated_results)) {
    val_data <- integrated_results$enhanced_validation
    if ("bootstrap_ci" %in% names(val_data)) {
      boot_data <- val_data$bootstrap_ci
      
      bootstrap_plot <- create_bootstrap_visualization(boot_data)
      plots$bootstrap <- bootstrap_plot
      
      ggsave(file.path(output_dir, "bootstrap_confidence_intervals.png"),
             bootstrap_plot, width = 12, height = 8, dpi = 300)
    }
  }
  
  return(plots)
}

#' Create Multiple Testing Correction Comparison Plot
#'
#' @param mtc_data Multiple testing correction data
#' @return ggplot object
create_mtc_comparison_plot <- function(mtc_data) {
  
  n_significant_uncorrected <- mtc_data$n_significant_uncorrected
  n_significant_corrected <- mtc_data$n_significant_corrected
  n_total <- mtc_data$n_tests
  
  # Create comparison data
  comparison_data <- data.frame(
    Stage = c("Before Correction", "After Correction"),
    Significant = c(n_significant_uncorrected, n_significant_corrected),
    Non_Significant = c(n_total - n_significant_uncorrected, n_total - n_significant_corrected),
    stringsAsFactors = FALSE
  )
  
  # Reshape for plotting
  comparison_long <- comparison_data %>%
    pivot_longer(cols = c(Significant, Non_Significant), 
                 names_to = "Status", values_to = "Count")
  
  # Create stacked bar plot
  p <- ggplot(comparison_long, aes(x = Stage, y = Count, fill = Status)) +
    geom_col(alpha = 0.8, color = "black", size = 0.5) +
    scale_fill_manual(values = c("Significant" = "#e74c3c", "Non_Significant" = "#3498db")) +
    labs(
      title = "Multiple Testing Correction Results",
      subtitle = paste("Method:", mtc_data$method, "; Total Tests:", n_total),
      x = "Analysis Stage",
      y = "Number of Tests",
      fill = "Statistical Significance"
    ) +
    theme_publication() +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(face = "bold", size = 12),
      legend.title = element_text(face = "bold", size = 11)
    )
  
  return(p)
}

#' Create Effect Size Visualization
#'
#' @param effect_data Effect size calculation data
#' @return ggplot object
create_effect_size_visualization <- function(effect_data) {
  
  # Extract effect sizes for visualization
  effect_sizes <- data.frame(
    Comparison = character(),
    Effect_Size = numeric(),
    Magnitude = character(),
    stringsAsFactors = FALSE
  )
  
  # Add apoptosis comparison
  if ("apoptosis_comparison" %in% names(effect_data)) {
    apoptosis_data <- effect_data$apoptosis_comparison
    if ("cohen_d_paired" %in% names(apoptosis_data)) {
      effect_sizes <- rbind(effect_sizes, data.frame(
        Comparison = "Apoptosis (Paired)",
        Effect_Size = apoptosis_data$cohen_d_paired,
        Magnitude = apoptosis_data$interpretation$magnitude,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Add SL score baseline comparison
  if ("sl_score_baseline" %in% names(effect_data)) {
    sl_data <- effect_data$sl_score_baseline
    if ("cohen_d_baseline" %in% names(sl_data)) {
      effect_sizes <- rbind(effect_sizes, data.frame(
        Comparison = "SL Score vs Baseline",
        Effect_Size = sl_data$cohen_d_baseline,
        Magnitude = sl_data$interpretation$magnitude,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  if (nrow(effect_sizes) == 0) {
    # Create placeholder plot
    p <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, label = "No effect size data available",
               size = 12, hjust = 0.5, vjust = 0.5) +
      theme_void() +
      labs(title = "Effect Size Analysis - No Data Available")
    return(p)
  }
  
  # Create effect size plot
  p <- ggplot(effect_sizes, aes(x = Comparison, y = Effect_Size, fill = Magnitude)) +
    geom_col(alpha = 0.8, color = "black", size = 0.5) +
    scale_fill_manual(values = c(
      "large" = "#e74c3c",
      "medium" = "#f39c12", 
      "small" = "#3498db",
      "negligible" = "#95a5a6"
    )) +
    geom_hline(yintercept = c(0.2, 0.5, 0.8), 
               linetype = c("dashed", "dashed", "dashed"),
               color = c("green", "orange", "red"), alpha = 0.5) +
    labs(
      title = "Effect Size Analysis (Cohen's d)",
      subtitle = "Threshold lines: 0.2 (small), 0.5 (medium), 0.8 (large)",
      x = "Comparison Type",
      y = "Cohen's d",
      fill = "Effect Magnitude"
    ) +
    theme_publication() +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.title = element_text(face = "bold", size = 11)
    )
  
  return(p)
}

#' Create Model Performance Assessment Plot
#'
#' @param perf_data Performance assessment data
#' @return ggplot object
create_performance_assessment_plot <- function(perf_data) {
  
  if (!("component_scores" %in% names(perf_data))) {
    p <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, label = "No performance data available",
               size = 12, hjust = 0.5, vjust = 0.5) +
      theme_void() +
      labs(title = "Model Performance Assessment - No Data Available")
    return(p)
  }
  
  # Extract component scores
  component_data <- data.frame(
    Component = names(perf_data$component_scores),
    Score = sapply(perf_data$component_scores, function(x) x$score),
    Level = sapply(perf_data$component_scores, function(x) x$level),
    stringsAsFactors = FALSE
  )
  
  # Create performance plot
  p <- ggplot(component_data, aes(x = Component, y = Score, fill = Level)) +
    geom_col(alpha = 0.8, color = "black", size = 0.5) +
    scale_fill_manual(values = c(
      "Excellent" = "#27ae60",
      "Good" = "#f39c12",
      "Fair" = "#e67e22", 
      "Poor" = "#e74c3c"
    )) +
    labs(
      title = "Model Performance Assessment",
      subtitle = paste("Overall Score:", round(perf_data$overall_score, 3), 
                     " | Overall Level:", perf_data$overall_level),
      x = "Performance Component",
      y = "Score",
      fill = "Performance Level"
    ) +
    theme_publication() +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.title = element_text(face = "bold", size = 11)
    )
  
  return(p)
}

#' Create Statistical Rigor Assessment Plot
#'
#' @param val_data Validation data
#' @return ggplot object
create_statistical_rigor_plot <- function(val_data) {
  
  if (!("significance_tests" %in% names(val_data))) {
    p <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, label = "No rigor assessment data available",
               size = 12, hjust = 0.5, vjust = 0.5) +
      theme_void() +
      labs(title = "Statistical Rigor Assessment - No Data Available")
    return(p)
  }
  
  # Extract test results
  test_data <- data.frame(
    Test = names(val_data$significance_tests),
    P_Value = sapply(val_data$significance_tests, function(x) x$p_value),
    Significant = sapply(val_data$significance_tests, function(x) x$significant),
    stringsAsFactors = FALSE
  )
  
  # Add significance stars
  test_data$Significance_Star <- ifelse(test_data$P_Value <= 0.001, "***",
                                       ifelse(test_data$P_Value <= 0.01, "**",
                                             ifelse(test_data$P_Value <= 0.05, "*", "ns")))
  
  # Create rigor assessment plot
  p <- ggplot(test_data, aes(x = Test, y = -log10(P_Value), fill = Significant)) +
    geom_col(alpha = 0.8, color = "black", size = 0.5) +
    scale_fill_manual(values = c("TRUE" = "#27ae60", "FALSE" = "#e74c3c")) +
    geom_hline(yintercept = -log10(0.05), linetype = "dashed", 
               color = "red", alpha = 0.7, size = 1) +
    labs(
      title = "Statistical Rigor Assessment",
      subtitle = "Red line: α = 0.05",
      x = "Statistical Test",
      y = "-log10(p-value)",
      fill = "Significant at α = 0.05"
    ) +
    theme_publication() +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.title = element_text(face = "bold", size = 11)
    )
  
  return(p)
}

#' Create Bootstrap Confidence Interval Visualization
#'
#' @param boot_data Bootstrap confidence interval data
#' @return ggplot object
create_bootstrap_visualization <- function(boot_data) {
  
  # Extract CI data for plotting
  ci_data <- data.frame(
    Metric = names(boot_data)[names(boot_data) %in% c("rmse", "mae", "r_squared")],
    Mean = sapply(boot_data[names(boot_data)[names(boot_data) %in% c("rmse", "mae", "r_squared")]], 
                 function(x) if(is.list(x)) x$mean else NA),
    CI_Lower = sapply(boot_data[names(boot_data)[names(boot_data) %in% c("rmse", "mae", "r_squared")]], 
                     function(x) if(is.list(x)) x$ci_lower else NA),
    CI_Upper = sapply(boot_data[names(boot_data)[names(boot_data) %in% c("rmse", "mae", "r_squared")]], 
                     function(x) if(is.list(x)) x$ci_upper else NA),
    stringsAsFactors = FALSE
  )
  
  # Remove rows with missing data
  ci_data <- ci_data[complete.cases(ci_data), ]
  
  if (nrow(ci_data) == 0) {
    p <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, label = "No bootstrap data available",
               size = 12, hjust = 0.5, vjust = 0.5) +
      theme_void() +
      labs(title = "Bootstrap Confidence Intervals - No Data Available")
    return(p)
  }
  
  # Create CI plot
  p <- ggplot(ci_data, aes(x = Metric, y = Mean)) +
    geom_point(size = 4, color = "steelblue") +
    geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), 
                 width = 0.2, size = 1, color = "steelblue") +
    labs(
      title = "Bootstrap 95% Confidence Intervals",
      subtitle = paste("Bootstrap samples:", boot_data[[1]]$n_bootstrap),
      x = "Validation Metric",
      y = "Metric Value",
      color = "95% CI"
    ) +
    theme_publication() +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      plot.subtitle = element_text(size = 12),
      axis.title = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 45, hjust = 1)
    )
  
  return(p)
}

#' Generate Comprehensive Publication Report
#'
#' Create complete publication-ready report with all statistical analyses
#'
#' @param integrated_results Integrated statistical analysis results
#' @param output_dir Output directory
#' @return List with report components
generate_comprehensive_publication_report <- function(integrated_results, 
                                                    output_dir = "publication_report") {
  
  # Create output directory
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  report_components <- list()
  
  # 1. Executive summary
  exec_summary <- create_executive_summary(integrated_results)
  report_components$executive_summary <- exec_summary
  
  # 2. Statistical methods section
  methods_section <- create_statistical_methods_section(integrated_results)
  report_components$methods_section <- methods_section
  
  # 3. Results section
  results_section <- create_results_section(integrated_results)
  report_components$results_section <- results_section
  
  # 4. Discussion section
  discussion_section <- create_discussion_section(integrated_results)
  report_components$discussion_section <- discussion_section
  
  # 5. Supplementary materials
  supp_materials <- create_supplementary_materials(integrated_results, output_dir)
  report_components$supplementary <- supp_materials
  
  # 6. Figure generation
  if (!is.null(integrated_results)) {
    plots <- create_statistical_integration_visualization(integrated_results, 
                                                        file.path(output_dir, "figures"))
    report_components$figures <- plots
  }
  
  # 7. Compile final report
  final_report <- compile_final_report(report_components, output_dir)
  report_components$final_report <- final_report
  
  return(report_components)
}

#' Create Executive Summary
#'
#' @param integrated_results Integrated analysis results
#' @return Executive summary text
create_executive_summary <- function(integrated_results) {
  
  summary_text <- "# Executive Summary\n\n"
  summary_text <- paste0(summary_text, 
                        "This report presents the comprehensive statistical analysis of the synthetic lethality QSP model.\n\n")
  
  # Add key findings
  if ("integration" %in% names(integrated_results)) {
    if ("key_findings" %in% names(integrated_results$integration)) {
      summary_text <- paste0(summary_text, "## Key Findings\n\n")
      for (finding in integrated_results$integration$key_findings) {
        summary_text <- paste0(summary_text, "- ", finding, "\n")
      }
      summary_text <- paste0(summary_text, "\n")
    }
  }
  
  # Add model performance
  if ("enhanced_validation" %in% names(integrated_results)) {
    val_data <- integrated_results$enhanced_validation
    if ("performance_assessment" %in% names(val_data)) {
      perf <- val_data$performance_assessment
      summary_text <- paste0(summary_text, 
                           "## Model Performance\n\n",
                           "Overall Performance Level: ", perf$overall_level, "\n",
                           "Performance Score: ", round(perf$overall_score, 3), "\n\n")
    }
  }
  
  # Add recommendations
  if ("integration" %in% names(integrated_results)) {
    if ("recommendations" %in% names(integrated_results$integration)) {
      summary_text <- paste0(summary_text, "## Recommendations\n\n")
      for (rec in integrated_results$integration$recommendations) {
        summary_text <- paste0(summary_text, "- ", rec, "\n")
      }
      summary_text <- paste0(summary_text, "\n")
    }
  }
  
  return(summary_text)
}

#' Create Statistical Methods Section
#'
#' @param integrated_results Integrated analysis results
#' @return Methods section text
create_statistical_methods_section <- function(integrated_results) {
  
  methods_text <- "# Statistical Methods\n\n"
  
  # Multiple testing correction
  if ("enhanced_synthetic_lethality" %in% names(integrated_results)) {
    sl_data <- integrated_results$enhanced_synthetic_lethality
    if ("multiple_testing" %in% names(sl_data)) {
      mtc <- sl_data$multiple_testing
      methods_text <- paste0(methods_text,
                           "## Multiple Testing Correction\n\n",
                           "Multiple testing correction was applied using the ", mtc$method, " method\n",
                           "to control for false discovery rate in ", mtc$n_tests, " hypothesis tests.\n\n")
    }
  }
  
  # Model validation
  methods_text <- paste0(methods_text,
                       "## Model Validation\n\n",
                       "Model validation was performed using cross-validation and statistical significance\n",
                       "testing to assess predictive performance and statistical rigor.\n\n")
  
  # Effect size analysis
  methods_text <- paste0(methods_text,
                       "## Effect Size Analysis\n\n",
                       "Effect sizes were calculated using Cohen's d to assess practical significance\n",
                       "of observed differences between treatment conditions.\n\n")
  
  return(methods_text)
}

#' Create Results Section
#'
#' @param integrated_results Integrated analysis results
#' @return Results section text
create_results_section <- function(integrated_results) {
  
  results_text <- "# Results\n\n"
  
  # Statistical significance
  if ("enhanced_synthetic_lethality" %in% names(integrated_results)) {
    sl_data <- integrated_results$enhanced_synthetic_lethality
    if ("multiple_testing" %in% names(sl_data)) {
      mtc <- sl_data$multiple_testing
      results_text <- paste0(results_text,
                           "## Statistical Significance\n\n",
                           "After multiple testing correction, ", mtc$n_significant_corrected, 
                           " out of ", mtc$n_tests, " tests remained significant\n",
                           "(", round(100 * mtc$n_significant_corrected / mtc$n_tests, 1), "%).\n\n")
    }
  }
  
  # Model performance
  if ("enhanced_validation" %in% names(integrated_results)) {
    val_data <- integrated_results$enhanced_validation
    if ("basic_metrics" %in% names(val_data)) {
      metrics <- val_data$basic_metrics
      results_text <- paste0(results_text,
                           "## Model Validation\n\n",
                           "Model validation yielded the following performance metrics:\n",
                           "- R²: ", round(metrics$r_squared, 3), "\n",
                           "- RMSE: ", round(metrics$rmse, 3), "\n",
                           "- Correlation: ", round(metrics$correlation, 3), "\n\n")
    }
  }
  
  return(results_text)
}

#' Create Discussion Section
#'
#' @param integrated_results Integrated analysis results
#' @return Discussion section text
create_discussion_section <- function(integrated_results) {
  
  discussion_text <- "# Discussion\n\n"
  discussion_text <- paste0(discussion_text,
                          "The integrated statistical analysis demonstrates the robustness and reliability\n",
                          "of the synthetic lethality QSP model through comprehensive validation.\n\n")
  
  # Add specific insights based on results
  if ("enhanced_validation" %in% names(integrated_results)) {
    val_data <- integrated_results$enhanced_validation
    if ("performance_assessment" %in% names(val_data)) {
      perf <- val_data$performance_assessment
      if (perf$overall_level %in% c("Excellent", "Good")) {
        discussion_text <- paste0(discussion_text,
                                "The model demonstrates ", tolower(perf$overall_level), 
                                " performance, indicating strong predictive capability\n",
                                "for synthetic lethality predictions.\n\n")
      }
    }
  }
  
  return(discussion_text)
}

#' Create Supplementary Materials
#'
#' @param integrated_results Integrated analysis results
#' @param output_dir Output directory
#' @return List of supplementary materials
create_supplementary_materials <- function(integrated_results, output_dir) {
  
  supp_materials <- list()
  
  # Statistical tables
  tables_dir <- file.path(output_dir, "tables")
  if (!dir.exists(tables_dir)) {
    dir.create(tables_dir, recursive = TRUE)
  }
  
  # Model validation table
  if ("enhanced_validation" %in% names(integrated_results)) {
    val_data <- integrated_results$enhanced_validation
    if ("basic_metrics" %in% names(val_data)) {
      metrics_table <- data.frame(
        Metric = c("R-squared", "RMSE", "MAE", "Correlation", "P-value"),
        Value = c(
          round(val_data$basic_metrics$r_squared, 3),
          round(val_data$basic_metrics$rmse, 3),
          round(val_data$basic_metrics$mae, 3),
          round(val_data$basic_metrics$correlation, 3),
          format(val_data$basic_metrics$correlation_p_value, scientific = TRUE)
        ),
        stringsAsFactors = FALSE
      )
      
      write_csv(metrics_table, file.path(tables_dir, "model_validation_metrics.csv"))
      supp_materials$validation_metrics <- metrics_table
    }
  }
  
  # Multiple testing correction table
  if ("enhanced_synthetic_lethality" %in% names(integrated_results)) {
    sl_data <- integrated_results$enhanced_synthetic_lethality
    if ("enhanced_screening_results" %in% names(sl_data)) {
      screening_data <- sl_data$enhanced_screening_results
      if (all(c("Drug", "Synthetic_Lethality_Score", "Corrected_pvalue") %in% names(screening_data))) {
        mtc_table <- screening_data %>%
          select(Drug, Synthetic_Lethality_Score, Corrected_pvalue) %>%
          rename(
            `Drug Name` = Drug,
            `SL Score` = Synthetic_Lethality_Score,
            `Corrected p-value` = Corrected_pvalue
          ) %>%
          arrange(`SL Score`) %>%
          head(20)  # Top 20 results
        
        write_csv(mtc_table, file.path(tables_dir, "multiple_testing_correction.csv"))
        supp_materials$mtc_table <- mtc_table
      }
    }
  }
  
  return(supp_materials)
}

#' Compile Final Report
#'
#' @param report_components Report components
#' @param output_dir Output directory
#' @return Path to final report
compile_final_report <- function(report_components, output_dir) {
  
  # Combine all sections
  full_report <- ""
  
  for (section_name in names(report_components)) {
    if (section_name != "final_report") {
      full_report <- paste0(full_report, report_components[[section_name]], "\n")
    }
  }
  
  # Add conclusion
  full_report <- paste0(full_report,
                       "# Conclusion\n\n",
                       "This comprehensive statistical analysis demonstrates the robustness and\n",
                       "reliability of the synthetic lethality QSP model through integrated\n",
                       "statistical validation approaches.\n\n")
  
  # Save report
  report_path <- file.path(output_dir, "comprehensive_statistical_report.md")
  writeLines(full_report, report_path)
  
  return(report_path)
}

# End of enhanced statistical integration methods
cat("Publication-Quality Visualization Framework loaded successfully (v1.0.0)\n")