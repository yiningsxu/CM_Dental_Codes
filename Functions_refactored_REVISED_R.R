# ============================================================================
# Refactored Analysis Functions (REVISED) - R translation
# Translated from Functions_refactored_REVISED(6).py
# ============================================================================

required_packages <- c("dplyr", "tidyr", "ggplot2", "splines", "MASS")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop(
    "Install the following R packages before running this script: ",
    paste(missing_packages, collapse = ", ")
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(splines)
})

timestamp <- format(Sys.Date(), "%Y%m%d")

# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

get_levels <- function(x) {
  if (is.factor(x)) {
    return(levels(x))
  }
  sort(unique(as.character(x[!is.na(x)])))
}

safe_numeric <- function(x) {
  suppressWarnings(as.numeric(x))
}

safe_sd <- function(x) {
  x <- safe_numeric(x)
  x <- x[!is.na(x)]
  if (length(x) <= 1) return(NA_real_)
  stats::sd(x)
}

format_p <- function(p, digits = 4) {
  if (is.na(p) || !is.finite(p)) return("N/A")
  if (p < 0.0001) return("<0.0001")
  sprintf(paste0("%.", digits, "f"), p)
}

format_p3 <- function(p) {
  if (is.na(p) || !is.finite(p)) return("")
  sprintf("%.3f", p)
}

format_mean_sd <- function(x, digits = 2, na_label = "N/A") {
  x <- safe_numeric(x)
  x <- x[!is.na(x)]
  if (length(x) == 0) return(na_label)
  m <- mean(x)
  s <- safe_sd(x)
  if (is.na(s)) return(sprintf(paste0("%.", digits, "f"), m))
  sprintf(paste0("%.", digits, "f ± %.", digits, "f"), m, s)
}

format_median_iqr <- function(x, digits = 1, na_label = "N/A") {
  x <- safe_numeric(x)
  x <- x[!is.na(x)]
  if (length(x) == 0) return(na_label)
  q <- stats::quantile(x, probs = c(0.25, 0.5, 0.75), na.rm = TRUE, names = FALSE, type = 7)
  sprintf(
    paste0("%.", digits, "f [%.", digits, "f-%.", digits, "f]"),
    q[2], q[1], q[3]
  )
}

fmt_num <- function(x, digits = 2) {
  if (is.na(x) || !is.finite(x)) return(NA)
  round(x, digits)
}

bind_rows_safe <- function(rows) {
  if (length(rows) == 0) return(data.frame())
  dplyr::bind_rows(lapply(rows, function(x) as.data.frame(x, stringsAsFactors = FALSE, check.names = FALSE)))
}

safe_chisq_p <- function(df, row_col, col_col) {
  if (!(row_col %in% names(df)) || !(col_col %in% names(df))) return(NA_real_)
  df_valid <- df[!is.na(df[[row_col]]) & !is.na(df[[col_col]]), , drop = FALSE]
  if (nrow(df_valid) == 0) return(NA_real_)
  tab <- table(df_valid[[row_col]], df_valid[[col_col]])
  if (nrow(tab) < 2 || ncol(tab) < 2) return(NA_real_)
  tryCatch(stats::chisq.test(tab)$p.value, error = function(e) NA_real_)
}

safe_kruskal_p <- function(groups) {
  groups <- groups[vapply(groups, function(g) length(g[!is.na(g)]) > 0, logical(1))]
  if (length(groups) < 2) return(NA_real_)
  values <- unlist(lapply(groups, function(g) safe_numeric(g[!is.na(g)])), use.names = FALSE)
  labels <- rep(seq_along(groups), vapply(groups, function(g) length(g[!is.na(g)]), integer(1)))
  if (length(unique(labels)) < 2 || length(values) == 0) return(NA_real_)
  tryCatch(stats::kruskal.test(values ~ as.factor(labels))$p.value, error = function(e) NA_real_)
}

add_fill0 <- function(a, b) {
  a <- safe_numeric(a)
  b <- safe_numeric(b)
  both_na <- is.na(a) & is.na(b)
  out <- ifelse(is.na(a), 0, a) + ifelse(is.na(b), 0, b)
  out[both_na] <- NA_real_
  out
}

# Dunn's post-hoc test using rank-based normal approximation.
# Returns symmetric matrices for adjusted and unadjusted p-values.
dunn_posthoc <- function(df, val_col, group_col, p_adjust = "bonferroni") {
  data <- df[!is.na(df[[val_col]]) & !is.na(df[[group_col]]), c(val_col, group_col), drop = FALSE]
  if (nrow(data) == 0) {
    return(list(adjusted = matrix(numeric(0)), unadjusted = matrix(numeric(0))))
  }

  data[[val_col]] <- safe_numeric(data[[val_col]])
  group_levels <- get_levels(data[[group_col]])
  group_levels <- group_levels[group_levels %in% as.character(unique(data[[group_col]]))]
  data[[group_col]] <- factor(as.character(data[[group_col]]), levels = group_levels)

  if (length(group_levels) < 2) {
    mat <- matrix(1, nrow = length(group_levels), ncol = length(group_levels), dimnames = list(group_levels, group_levels))
    return(list(adjusted = mat, unadjusted = mat))
  }

  ranks <- rank(data[[val_col]], ties.method = "average")
  n_total <- length(ranks)
  tie_counts <- table(data[[val_col]])
  tie_adjustment <- sum(tie_counts^3 - tie_counts) / (12 * max(n_total - 1, 1))
  s2 <- (n_total * (n_total + 1) / 12) - tie_adjustment

  mean_ranks <- tapply(ranks, data[[group_col]], mean)
  group_n <- table(data[[group_col]])

  unadj <- matrix(1, nrow = length(group_levels), ncol = length(group_levels), dimnames = list(group_levels, group_levels))
  pairs <- utils::combn(group_levels, 2, simplify = FALSE)

  for (pair in pairs) {
    g1 <- pair[1]
    g2 <- pair[2]
    if (is.na(mean_ranks[g1]) || is.na(mean_ranks[g2]) || group_n[g1] == 0 || group_n[g2] == 0) next
    z <- abs(mean_ranks[g1] - mean_ranks[g2]) / sqrt(s2 * (1 / group_n[g1] + 1 / group_n[g2]))
    p <- 2 * stats::pnorm(-abs(z))
    unadj[g1, g2] <- p
    unadj[g2, g1] <- p
  }

  adjusted <- unadj
  if (!is.null(p_adjust) && p_adjust == "bonferroni") {
    m <- length(pairs)
    adjusted <- pmin(unadj * m, 1)
    diag(adjusted) <- 1
  }

  list(adjusted = adjusted, unadjusted = unadj)
}

p_value_to_numeric <- function(x) {
  if (is.na(x)) return(NA_real_)
  x <- as.character(x)
  if (startsWith(x, "<")) return(safe_numeric(sub("<", "", x)))
  safe_numeric(x)
}

# -----------------------------------------------------------------------------
# CSV value-count summaries
# -----------------------------------------------------------------------------

save_value_counts_summary <- function(df, output_path, exclude_cols = NULL) {
  if (is.null(exclude_cols)) exclude_cols <- character(0)
  results <- list()

  for (col in names(df)) {
    if (col %in% exclude_cols) next
    values <- df[[col]]
    value_counts <- table(ifelse(is.na(values), "NA", as.character(values)), useNA = "ifany")
    for (val in names(value_counts)) {
      results[[length(results) + 1]] <- list(
        Column = col,
        Value = val,
        Count = as.integer(value_counts[[val]])
      )
    }
  }

  results_df <- bind_rows_safe(results)
  utils::write.csv(results_df, output_path, row.names = FALSE)
  message("Summary saved to ", output_path)
  invisible(NULL)
}

# -----------------------------------------------------------------------------
# Table 1: demographics by abuse type
# -----------------------------------------------------------------------------

create_table1_demographics <- function(df) {
  df_local <- df
  abuse_types <- get_levels(df_local$abuse)
  results <- list()

  total_row <- list(Variable = "Total N", Category = "")
  for (abuse in abuse_types) {
    total_row[[abuse]] <- as.character(sum(df_local$abuse == abuse, na.rm = TRUE))
  }
  total_row[["Total"]] <- as.character(nrow(df_local))
  total_row[["p-value"]] <- ""
  results[[length(results) + 1]] <- total_row

  sex_header <- list(Variable = "Sex", Category = "")
  for (abuse in abuse_types) sex_header[[abuse]] <- ""
  sex_header[["Total"]] <- ""
  sex_header[["p-value"]] <- ""
  results[[length(results) + 1]] <- sex_header

  p_sex <- safe_chisq_p(df_local, "abuse", "sex")
  for (sex in c("Male", "Female")) {
    row <- list(Variable = "", Category = paste0("  ", sex))
    for (abuse in abuse_types) {
      subset <- df_local[df_local$abuse == abuse, , drop = FALSE]
      n <- sum(subset$sex == sex, na.rm = TRUE)
      total_abuse <- nrow(subset)
      pct <- ifelse(total_abuse > 0, n / total_abuse * 100, 0)
      row[[abuse]] <- sprintf("%d (%.1f%%)", n, pct)
    }
    total_n <- sum(df_local$sex == sex, na.rm = TRUE)
    total_pct <- ifelse(nrow(df_local) > 0, total_n / nrow(df_local) * 100, 0)
    row[["Total"]] <- sprintf("%d (%.1f%%)", total_n, total_pct)
    row[["p-value"]] <- ifelse(sex == "Male" && !is.na(p_sex), format_p3(p_sex), "")
    results[[length(results) + 1]] <- row
  }

  age_row <- list(Variable = "Age (years)", Category = "Mean ± SD")
  for (abuse in abuse_types) {
    subset <- df_local[df_local$abuse == abuse, "age_year"]
    age_row[[abuse]] <- format_mean_sd(subset, digits = 1)
  }
  age_row[["Total"]] <- format_mean_sd(df_local$age_year, digits = 1)
  groups <- lapply(abuse_types, function(abuse) df_local[df_local$abuse == abuse, "age_year"])
  p_age <- safe_kruskal_p(groups)
  age_row[["p-value"]] <- ifelse(is.na(p_age), "N/A", format_p3(p_age))
  results[[length(results) + 1]] <- age_row

  age_median_row <- list(Variable = "", Category = "Median [IQR]")
  for (abuse in abuse_types) {
    subset <- df_local[df_local$abuse == abuse, "age_year"]
    age_median_row[[abuse]] <- format_median_iqr(subset, digits = 0)
  }
  age_median_row[["Total"]] <- format_median_iqr(df_local$age_year, digits = 0)
  age_median_row[["p-value"]] <- ""
  results[[length(results) + 1]] <- age_median_row

  if ("age_group" %in% names(df_local)) {
    age_group_header <- list(Variable = "Age Group", Category = "")
    for (abuse in abuse_types) age_group_header[[abuse]] <- ""
    age_group_header[["Total"]] <- ""
    age_group_header[["p-value"]] <- ""
    results[[length(results) + 1]] <- age_group_header

    p_age_grp <- safe_chisq_p(df_local, "abuse", "age_group")
    age_group_levels <- get_levels(df_local$age_group)
    first_group <- TRUE

    for (age_grp in age_group_levels) {
      row <- list(Variable = "", Category = paste0("  ", age_grp))
      for (abuse in abuse_types) {
        subset <- df_local[df_local$abuse == abuse, , drop = FALSE]
        n <- sum(subset$age_group == age_grp, na.rm = TRUE)
        total_abuse <- nrow(subset)
        pct <- ifelse(total_abuse > 0, n / total_abuse * 100, 0)
        row[[abuse]] <- sprintf("%d (%.1f%%)", n, pct)
      }
      total_n <- sum(df_local$age_group == age_grp, na.rm = TRUE)
      total_pct <- ifelse(nrow(df_local) > 0, total_n / nrow(df_local) * 100, 0)
      row[["Total"]] <- sprintf("%d (%.1f%%)", total_n, total_pct)
      row[["p-value"]] <- ifelse(first_group && !is.na(p_age_grp), format_p3(p_age_grp), "")
      first_group <- FALSE
      results[[length(results) + 1]] <- row
    }
  }

  bind_rows_safe(results)
}

# -----------------------------------------------------------------------------
# Table 1.1: demographics by dentition period and abuse type
# -----------------------------------------------------------------------------

create_table1_1_demographics_by_dentition <- function(df) {
  df_local <- df
  abuse_types <- get_levels(df_local$abuse)
  dentition_order <- c("primary_dentition", "mixed_dentition", "permanent_dentition")
  results <- list()

  for (dent_type in dentition_order) {
    df_dent <- df_local[df_local$dentition_type == dent_type, , drop = FALSE]
    if (nrow(df_dent) == 0) next

    age_total <- safe_numeric(df_dent$age_year)
    age_total <- age_total[!is.na(age_total)]
    if (length(age_total) > 0) {
      q <- stats::quantile(age_total, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      sd_val <- safe_sd(age_total)
      results[[length(results) + 1]] <- list(
        Dentition_Period = dent_type,
        Group = "Total",
        N = length(age_total),
        Mean = round(mean(age_total), 2),
        SD = round(sd_val, 2),
        Median = round(stats::median(age_total), 2),
        IQR = sprintf("%.2f-%.2f", q[1], q[2]),
        Min = round(min(age_total), 2),
        Max = round(max(age_total), 2),
        `Mean±SD` = ifelse(is.na(sd_val), sprintf("%.2f", mean(age_total)), sprintf("%.2f ± %.2f", mean(age_total), sd_val)),
        `Median[IQR]` = sprintf("%.1f [%.1f-%.1f]", stats::median(age_total), q[1], q[2]),
        `Min-Max` = sprintf("%.1f-%.1f", min(age_total), max(age_total))
      )
    }

    for (abuse in abuse_types) {
      age_sub <- safe_numeric(df_dent[df_dent$abuse == abuse, "age_year"])
      age_sub <- age_sub[!is.na(age_sub)]
      if (length(age_sub) == 0) next
      q <- stats::quantile(age_sub, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      sd_val <- safe_sd(age_sub)
      results[[length(results) + 1]] <- list(
        Dentition_Period = dent_type,
        Group = abuse,
        N = length(age_sub),
        Mean = round(mean(age_sub), 2),
        SD = round(sd_val, 2),
        Median = round(stats::median(age_sub), 2),
        IQR = sprintf("%.2f-%.2f", q[1], q[2]),
        Min = round(min(age_sub), 2),
        Max = round(max(age_sub), 2),
        `Mean±SD` = ifelse(is.na(sd_val), sprintf("%.2f", mean(age_sub)), sprintf("%.2f ± %.2f", mean(age_sub), sd_val)),
        `Median[IQR]` = sprintf("%.1f [%.1f-%.1f]", stats::median(age_sub), q[1], q[2]),
        `Min-Max` = sprintf("%.1f-%.1f", min(age_sub), max(age_sub))
      )
    }
  }

  bind_rows_safe(results)
}

# -----------------------------------------------------------------------------
# Table 2: descriptive statistics for oral health
# -----------------------------------------------------------------------------

create_table2_oral_health_descriptive <- function(df) {
  abuse_types <- get_levels(df$abuse)
  ratio_vars <- c("Care_Index", "UTN_Score")

  continuous_vars <- list(
    c("DMFT_Index", "DMFT Index (Total)"),
    c("decayed_total", "Decayed Total (D+d)"),
    c("missing_total", "Missing Total (M+m)"),
    c("filled_total", "Filled Total (F+f)"),
    c("Perm_DMFT", "Permanent DMFT"),
    c("Baby_DMFT", "Primary dmft"),
    c("Perm_D", "Permanent D (Decayed)"),
    c("Perm_M", "Permanent M (Missing)"),
    c("Perm_F", "Permanent F (Filled)"),
    c("Baby_d", "Primary d (decayed)"),
    c("Baby_m", "Primary m (missing)"),
    c("Baby_f", "Primary f (filled)"),
    c("C0_Count", "C0 (Incipient Caries)"),
    c("Healthy_Rate", "Healthy Teeth Rate (%)"),
    c("Care_Index", "Care Index (%) (DMFT_Index>0 only)"),
    c("UTN_Score", "Untreated Caries Rate (%) (DMFT_Index>0 only)"),
    c("Trauma_Count", "Dental Trauma Count"),
    c("RDT_Count", "Retained Deciduous Teeth")
  )

  filtered_df_for_var <- function(dfx, var_name) {
    if (var_name %in% ratio_vars && "DMFT_Index" %in% names(dfx)) {
      return(dfx[dfx$DMFT_Index > 0 & !is.na(dfx$DMFT_Index), , drop = FALSE])
    }
    dfx
  }

  results_continuous <- list()

  for (var_pair in continuous_vars) {
    var_name <- var_pair[1]
    var_label <- var_pair[2]
    if (!(var_name %in% names(df))) next

    row <- list(Variable = var_label)
    for (abuse in abuse_types) {
      df_sub <- filtered_df_for_var(df[df$abuse == abuse, , drop = FALSE], var_name)
      subset <- safe_numeric(df_sub[[var_name]])
      subset <- subset[!is.na(subset)]
      if (length(subset) > 0) {
        row[[paste0(abuse, "_Mean_SD")]] <- format_mean_sd(subset, digits = 2)
        row[[paste0(abuse, "_Median_IQR")]] <- format_median_iqr(subset, digits = 1)
      } else {
        row[[paste0(abuse, "_Mean_SD")]] <- "N/A"
        row[[paste0(abuse, "_Median_IQR")]] <- "N/A"
      }
    }

    df_total <- filtered_df_for_var(df, var_name)
    total <- safe_numeric(df_total[[var_name]])
    total <- total[!is.na(total)]
    if (length(total) > 0) {
      row[["Total_Mean_SD"]] <- format_mean_sd(total, digits = 2)
      row[["Total_Median_IQR"]] <- format_median_iqr(total, digits = 1)
    } else {
      row[["Total_Mean_SD"]] <- "N/A"
      row[["Total_Median_IQR"]] <- "N/A"
    }

    groups <- lapply(abuse_types, function(abuse) {
      df_sub <- filtered_df_for_var(df[df$abuse == abuse, , drop = FALSE], var_name)
      safe_numeric(df_sub[[var_name]])[!is.na(safe_numeric(df_sub[[var_name]]))]
    })
    p_val <- safe_kruskal_p(groups)
    row[["p-value"]] <- ifelse(is.na(p_val), "N/A", format_p(p_val))
    results_continuous[[length(results_continuous) + 1]] <- row
  }

  categorical_vars <- list(
    c("gingivitis", "Gingivitis"),
    c("needTOBEtreated", "Treatment Need"),
    c("occlusalRelationship", "Occlusal Relationship"),
    c("OralCleanStatus", "Oral Hygiene Status"),
    c("habits", "Oral Habits")
  )

  results_categorical <- list()

  for (var_pair in categorical_vars) {
    var_name <- var_pair[1]
    var_label <- var_pair[2]
    if (!(var_name %in% names(df))) next

    header_row <- list(Variable = var_label, Category = "")
    for (abuse in abuse_types) {
      header_row[[paste0(abuse, "_n")]] <- ""
      header_row[[paste0(abuse, "_%")]] <- ""
    }
    header_row[["Total_n"]] <- ""
    header_row[["Total_%"]] <- ""
    header_row[["p-value"]] <- ""
    results_categorical[[length(results_categorical) + 1]] <- header_row

    p_val <- safe_chisq_p(df, "abuse", var_name)
    categories <- get_levels(df[[var_name]])
    first_cat <- TRUE

    for (cat in categories) {
      row <- list(Variable = "", Category = paste0("  ", cat))
      for (abuse in abuse_types) {
        subset <- df[df$abuse == abuse & !is.na(df[[var_name]]), , drop = FALSE]
        n <- sum(df$abuse == abuse & df[[var_name]] == cat, na.rm = TRUE)
        total_abuse <- nrow(subset)
        pct <- ifelse(total_abuse > 0, n / total_abuse * 100, 0)
        row[[paste0(abuse, "_n")]] <- n
        row[[paste0(abuse, "_%")]] <- sprintf("%.1f", pct)
      }
      total_n <- sum(df[[var_name]] == cat, na.rm = TRUE)
      total_valid <- sum(!is.na(df[[var_name]]))
      total_pct <- ifelse(total_valid > 0, total_n / total_valid * 100, 0)
      row[["Total_n"]] <- total_n
      row[["Total_%"]] <- sprintf("%.1f", total_pct)
      row[["p-value"]] <- ifelse(first_cat && !is.na(p_val), format_p(p_val), "")
      first_cat <- FALSE
      results_categorical[[length(results_categorical) + 1]] <- row
    }
  }

  list(
    continuous = bind_rows_safe(results_continuous),
    categorical = bind_rows_safe(results_categorical)
  )
}

# -----------------------------------------------------------------------------
# Table 3: statistical comparisons and post-hoc tests
# -----------------------------------------------------------------------------

create_table3_statistical_comparisons <- function(df) {
  abuse_types <- get_levels(df$abuse)

  continuous_vars <- c(
    "DMFT_Index", "decayed_total", "missing_total", "filled_total",
    "Perm_DMFT", "Baby_DMFT",
    "Perm_D", "Perm_M", "Perm_F",
    "Baby_d", "Baby_m", "Baby_f",
    "C0_Count", "Healthy_Rate", "Care_Index",
    "UTN_Score", "Trauma_Count", "DMFT_C0", "Perm_DMFT_C0", "Baby_DMFT_C0"
  )
  ratio_vars <- c("Care_Index", "UTN_Score")

  df_for_var <- function(dfx, var) {
    if (var %in% ratio_vars && "DMFT_Index" %in% names(dfx)) {
      return(dfx[dfx$DMFT_Index > 0 & !is.na(dfx$DMFT_Index), , drop = FALSE])
    }
    dfx
  }

  present_vars <- continuous_vars[continuous_vars %in% names(df)]
  overall_results <- list()

  for (var in present_vars) {
    df_var <- df_for_var(df, var)
    df_var <- df_var[!is.na(df_var[[var]]) & !is.na(df_var$abuse), , drop = FALSE]
    groups <- lapply(abuse_types, function(abuse) {
      safe_numeric(df_var[df_var$abuse == abuse, var])
    })
    groups <- lapply(groups, function(g) g[!is.na(g)])
    groups <- groups[vapply(groups, length, integer(1)) > 0]
    if (length(groups) < 2) next

    row <- list(Variable = var, Test = "Kruskal-Wallis")
    total_data <- safe_numeric(df_var[[var]])
    total_data <- total_data[!is.na(total_data)]
    if (length(total_data) > 0) {
      row[["Total_Mean_SD"]] <- format_mean_sd(total_data, digits = 2)
      row[["Total_Median_IQR"]] <- format_median_iqr(total_data, digits = 1)
    } else {
      row[["Total_Mean_SD"]] <- "N/A"
      row[["Total_Median_IQR"]] <- "N/A"
    }

    for (abuse in abuse_types) {
      subset <- safe_numeric(df_var[df_var$abuse == abuse, var])
      subset <- subset[!is.na(subset)]
      if (length(subset) > 0) {
        row[[paste0(abuse, "_Mean_SD")]] <- format_mean_sd(subset, digits = 2)
        row[[paste0(abuse, "_Median_IQR")]] <- format_median_iqr(subset, digits = 1)
      } else {
        row[[paste0(abuse, "_Mean_SD")]] <- "N/A"
        row[[paste0(abuse, "_Median_IQR")]] <- "N/A"
      }
    }

    p_kw <- safe_kruskal_p(groups)
    h_stat <- tryCatch({
      values <- unlist(groups, use.names = FALSE)
      labels <- rep(seq_along(groups), vapply(groups, length, integer(1)))
      stats::kruskal.test(values ~ as.factor(labels))$statistic[[1]]
    }, error = function(e) NA_real_)

    row[["Statistic"]] <- ifelse(is.na(h_stat), "N/A", sprintf("%.3f", h_stat))
    row[["p-value"]] <- ifelse(is.na(p_kw), "N/A", format_p(p_kw))
    row[["Significant"]] <- ifelse(!is.na(p_kw) && p_kw < 0.05, "Yes", ifelse(is.na(p_kw), "N/A", "No"))
    overall_results[[length(overall_results) + 1]] <- row
  }

  overall_df <- bind_rows_safe(overall_results)
  posthoc_results <- list()
  tidy_posthoc_pairwise <- list()

  for (var in present_vars) {
    if (nrow(overall_df) == 0) next
    is_sig <- overall_df$Significant[overall_df$Variable == var]
    if (length(is_sig) == 0 || is_sig[1] != "Yes") next

    df_var <- df_for_var(df, var)
    df_var <- df_var[!is.na(df_var[[var]]) & !is.na(df_var$abuse), , drop = FALSE]
    df_var$`_rank` <- rank(safe_numeric(df_var[[var]]), ties.method = "average")
    mean_ranks <- tapply(df_var$`_rank`, df_var$abuse, mean)

    dunn <- tryCatch(dunn_posthoc(df_var, val_col = var, group_col = "abuse", p_adjust = "bonferroni"), error = function(e) NULL)
    if (is.null(dunn)) next
    dunn_adj <- dunn$adjusted
    dunn_unadj <- dunn$unadjusted

    for (i in seq_along(abuse_types)) {
      if (i == length(abuse_types)) next
      for (j in (i + 1):length(abuse_types)) {
        abuse1 <- abuse_types[i]
        abuse2 <- abuse_types[j]
        if (!(abuse1 %in% rownames(dunn_adj)) || !(abuse2 %in% colnames(dunn_adj))) next

        p_adj <- as.numeric(dunn_adj[abuse1, abuse2])
        p_unadj <- as.numeric(dunn_unadj[abuse1, abuse2])
        group1_vals <- safe_numeric(df_var[df_var$abuse == abuse1, var])
        group2_vals <- safe_numeric(df_var[df_var$abuse == abuse2, var])
        group1_vals <- group1_vals[!is.na(group1_vals)]
        group2_vals <- group2_vals[!is.na(group2_vals)]
        g1_n <- length(group1_vals)
        g2_n <- length(group2_vals)

        if (g1_n == 0 || g2_n == 0) next

        g1_mean <- mean(group1_vals)
        g2_mean <- mean(group2_vals)
        g1_sd <- safe_sd(group1_vals)
        g2_sd <- safe_sd(group2_vals)
        g1_median <- stats::median(group1_vals)
        g2_median <- stats::median(group2_vals)
        g1_q <- stats::quantile(group1_vals, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
        g2_q <- stats::quantile(group2_vals, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
        g1_mean_rank <- mean_ranks[abuse1]
        g2_mean_rank <- mean_ranks[abuse2]

        g1_mean_sd_str <- ifelse(is.na(g1_sd), sprintf("%.2f", g1_mean), sprintf("%.2f ± %.2f", g1_mean, g1_sd))
        g2_mean_sd_str <- ifelse(is.na(g2_sd), sprintf("%.2f", g2_mean), sprintf("%.2f ± %.2f", g2_mean, g2_sd))
        g1_median_iqr_str <- sprintf("%.1f [%.1f-%.1f]", g1_median, g1_q[1], g1_q[2])
        g2_median_iqr_str <- sprintf("%.1f [%.1f-%.1f]", g2_median, g2_q[1], g2_q[2])

        posthoc_results[[length(posthoc_results) + 1]] <- list(
          Variable = var,
          Group1 = abuse1,
          Group2 = abuse2,
          Comparison = paste0(abuse1, " vs ", abuse2),
          Group1_n = g1_n,
          Group2_n = g2_n,
          Group1_Mean = round(g1_mean, 2),
          Group2_Mean = round(g2_mean, 2),
          Group1_SD = round(g1_sd, 2),
          Group2_SD = round(g2_sd, 2),
          Group1_Median = round(g1_median, 2),
          Group2_Median = round(g2_median, 2),
          Group1_IQR = sprintf("%.1f-%.1f", g1_q[1], g1_q[2]),
          Group2_IQR = sprintf("%.1f-%.1f", g2_q[1], g2_q[2]),
          Group1_Mean_SD = g1_mean_sd_str,
          Group2_Mean_SD = g2_mean_sd_str,
          Group1_Median_IQR = g1_median_iqr_str,
          Group2_Median_IQR = g2_median_iqr_str,
          Group1_Mean_Rank = round(g1_mean_rank, 2),
          Group2_Mean_Rank = round(g2_mean_rank, 2),
          `p-value (unadjusted)` = format_p(p_unadj),
          `p-value (adjusted)` = format_p(p_adj),
          Significant = ifelse(p_adj < 0.05, "Yes", "No")
        )

        tidy_posthoc_pairwise[[length(tidy_posthoc_pairwise) + 1]] <- list(
          variable = var,
          group1 = abuse1,
          group2 = abuse2,
          group1_n = g1_n,
          group2_n = g2_n,
          group1_mean = g1_mean,
          group2_mean = g2_mean,
          group1_sd = g1_sd,
          group2_sd = g2_sd,
          group1_median = g1_median,
          group2_median = g2_median,
          group1_q1 = g1_q[1],
          group1_q3 = g1_q[2],
          group2_q1 = g2_q[1],
          group2_q3 = g2_q[2],
          group1_mean_sd_str = g1_mean_sd_str,
          group2_mean_sd_str = g2_mean_sd_str,
          group1_median_iqr_str = g1_median_iqr_str,
          group2_median_iqr_str = g2_median_iqr_str,
          group1_mean_rank = g1_mean_rank,
          group2_mean_rank = g2_mean_rank,
          p_unadjusted = p_unadj,
          p_adjusted = p_adj,
          significant = p_adj < 0.05,
          analysis_type = "Table 3: Overall"
        )
      }
    }
  }

  pairwise_results <- list()
  abuse_pairs <- utils::combn(abuse_types, 2, simplify = FALSE)
  n_comparisons <- length(abuse_pairs) * max(length(present_vars), 1)
  bonferroni_threshold <- ifelse(n_comparisons > 0, 0.05 / n_comparisons, 0.05)

  for (var in present_vars) {
    df_var <- df_for_var(df, var)
    for (pair in abuse_pairs) {
      abuse1 <- pair[1]
      abuse2 <- pair[2]
      group1 <- safe_numeric(df_var[df_var$abuse == abuse1, var])
      group2 <- safe_numeric(df_var[df_var$abuse == abuse2, var])
      group1 <- group1[!is.na(group1)]
      group2 <- group2[!is.na(group2)]
      if (length(group1) == 0 || length(group2) == 0) next

      res <- tryCatch(stats::wilcox.test(group1, group2, alternative = "two.sided", exact = FALSE), error = function(e) NULL)
      if (is.null(res)) next
      u_stat <- as.numeric(res$statistic)
      p_val <- res$p.value
      r <- 1 - (2 * u_stat) / (length(group1) * length(group2))

      pairwise_results[[length(pairwise_results) + 1]] <- list(
        Variable = var,
        Group1 = abuse1,
        Group2 = abuse2,
        Group1_Median = sprintf("%.1f", stats::median(group1)),
        Group2_Median = sprintf("%.1f", stats::median(group2)),
        U_Statistic = sprintf("%.0f", u_stat),
        `p-value` = format_p(p_val),
        Effect_Size_r = sprintf("%.3f", r),
        Significant_Bonferroni = ifelse(p_val < bonferroni_threshold, "Yes", "No")
      )
    }
  }

  list(
    overall = overall_df,
    posthoc = bind_rows_safe(posthoc_results),
    pairwise = bind_rows_safe(pairwise_results),
    tidy_posthoc_pairwise = bind_rows_safe(tidy_posthoc_pairwise)
  )
}

# -----------------------------------------------------------------------------
# Firth penalized logistic regression fallback and multivariable models
# -----------------------------------------------------------------------------

_firth_logit <- function(X, y, maxiter = 100, tol = 1e-8) {
  X <- as.matrix(X)
  y <- as.numeric(y)
  n <- nrow(X)
  p <- ncol(X)
  beta <- rep(0, p)

  sigmoid <- function(z) 1 / (1 + exp(-z))
  converged <- FALSE

  for (iter in seq_len(maxiter)) {
    eta <- as.vector(X %*% beta)
    mu <- sigmoid(eta)
    W <- pmax(mu * (1 - mu), 1e-9)
    XtW <- t(X) * W
    I <- XtW %*% X
    I_inv <- tryCatch(solve(I), error = function(e) MASS::ginv(I))
    h <- rowSums((X %*% I_inv) * X) * W
    a <- (0.5 - mu) * h
    z <- eta + (y - mu + a) / W
    beta_new <- I_inv %*% (t(X) %*% (W * z))
    beta_new <- as.numeric(beta_new)
    if (max(abs(beta_new - beta), na.rm = TRUE) < tol) {
      beta <- beta_new
      converged <- TRUE
      break
    }
    beta <- beta_new
  }

  eta <- as.vector(X %*% beta)
  mu <- sigmoid(eta)
  W <- pmax(mu * (1 - mu), 1e-9)
  I <- (t(X) * W) %*% X
  cov <- tryCatch(solve(I), error = function(e) MASS::ginv(I))
  se <- sqrt(diag(cov))
  list(beta = beta, se = se, converged = converged)
}

_fit_pairwise_logit <- function(
  df_model,
  outcome_var,
  use_age_spline = TRUE,
  age_spline_df = 4,
  extra_terms = NULL,
  id_col = NULL,
  force_firth = FALSE
) {
  if (is.null(extra_terms)) extra_terms <- character(0)

  age_term <- if (use_age_spline) {
    sprintf("splines::ns(age_year, df = %d)", age_spline_df)
  } else {
    "age_year"
  }

  rhs_terms <- c(age_term, "sex_male", "comparison", extra_terms)
  formula <- stats::as.formula(paste(outcome_var, "~", paste(rhs_terms, collapse = " + ")))

  # Standard MLE first unless Firth is forced.
  if (!force_firth) {
    mle_fit <- tryCatch(
      stats::glm(formula, data = df_model, family = stats::binomial()),
      error = function(e) NULL,
      warning = function(w) suppressWarnings(stats::glm(formula, data = df_model, family = stats::binomial()))
    )

    if (!is.null(mle_fit)) {
      coefs <- tryCatch(summary(mle_fit)$coefficients, error = function(e) NULL)

      if (!is.null(id_col) && id_col %in% names(df_model) && length(unique(df_model[[id_col]])) < nrow(df_model)) {
        if (requireNamespace("sandwich", quietly = TRUE) && requireNamespace("lmtest", quietly = TRUE)) {
          robust <- tryCatch(
            lmtest::coeftest(mle_fit, vcov. = sandwich::vcovCL(mle_fit, cluster = df_model[[id_col]])),
            error = function(e) NULL
          )
          if (!is.null(robust)) coefs <- robust
        }
      }

      if (!is.null(coefs) && "comparison" %in% rownames(coefs)) {
        beta <- as.numeric(coefs["comparison", "Estimate"])
        se <- as.numeric(coefs["comparison", "Std. Error"])
        p_val <- as.numeric(coefs["comparison", ncol(coefs)])
        if (all(is.finite(c(beta, se, p_val))) && abs(beta) < 50 && se < 50) {
          return(list(beta = beta, se = se, p_value = p_val, model = "Logit (MLE)"))
        }
      }
    }
  }

  # Prefer logistf when available; otherwise use the matrix implementation above.
  if (requireNamespace("logistf", quietly = TRUE)) {
    firth_fit <- tryCatch(logistf::logistf(formula, data = df_model), error = function(e) NULL)
    if (!is.null(firth_fit) && "comparison" %in% names(stats::coef(firth_fit))) {
      beta <- as.numeric(stats::coef(firth_fit)["comparison"])
      se <- as.numeric(sqrt(diag(firth_fit$var))["comparison"])
      p_val <- if ("comparison" %in% names(firth_fit$prob)) as.numeric(firth_fit$prob["comparison"]) else NA_real_
      return(list(beta = beta, se = se, p_value = p_val, model = "Logit (Firth)"))
    }
  }

  X <- stats::model.matrix(formula, data = df_model)
  y <- safe_numeric(df_model[[outcome_var]])
  firth <- _firth_logit(X, y, maxiter = 200, tol = 1e-8)
  j <- which(colnames(X) == "comparison")
  if (length(j) == 0) j <- ncol(X)
  beta <- as.numeric(firth$beta[j])
  se <- as.numeric(firth$se[j])
  z <- ifelse(!is.na(se) && se > 0, beta / se, NA_real_)
  p_val <- ifelse(!is.na(z), 2 * (1 - stats::pnorm(abs(z))), NA_real_)
  model_name <- ifelse(firth$converged, "Logit (Firth)", "Logit (Firth; not converged)")

  list(beta = beta, se = se, p_value = p_val, model = model_name)
}

create_table4_multivariate_analysis <- function(
  df,
  reference_category = "Physical Abuse",
  comparison_categories = NULL,
  use_age_spline = TRUE,
  age_spline_df = 4,
  add_year_fe = TRUE,
  year_col = "year",
  examiner_col = NULL,
  add_covariates = NULL,
  id_col = NULL,
  stratify_by = NULL,
  strata_order = NULL,
  min_n = 50,
  force_firth = FALSE
) {
  results <- list()
  df_analysis <- df

  if ("sex" %in% names(df_analysis)) {
    df_analysis$sex_male <- as.integer(df_analysis$sex == "Male")
  }

  if (is.null(comparison_categories)) {
    comparison_categories <- c("Neglect", "Emotional Abuse", "Sexual Abuse")
  }

  outcomes <- list(
    c("has_caries", "Caries Experience (>0)"),
    c("has_untreated_caries", "Untreated Caries")
  )

  if ("gingivitis" %in% names(df_analysis)) {
    df_analysis$gingivitis_binary <- as.integer(df_analysis$gingivitis == "Gingivitis")
    outcomes[[length(outcomes) + 1]] <- c("gingivitis_binary", "Gingivitis")
  }

  if ("needTOBEtreated" %in% names(df_analysis)) {
    df_analysis$treatment_need <- as.integer(df_analysis$needTOBEtreated == "Treatment Required")
    outcomes[[length(outcomes) + 1]] <- c("treatment_need", "Treatment Need")
  }

  if (is.null(add_covariates)) add_covariates <- character(0)

  if (is.null(stratify_by)) {
    strata <- list(list(label = "Overall", data = df_analysis))
  } else {
    if (is.null(strata_order)) {
      vals <- sort(unique(as.character(df_analysis[[stratify_by]][!is.na(df_analysis[[stratify_by]])])))
    } else {
      vals <- strata_order[strata_order %in% as.character(unique(df_analysis[[stratify_by]]))]
    }
    strata <- lapply(vals, function(v) list(label = as.character(v), data = df_analysis[df_analysis[[stratify_by]] == v, , drop = FALSE]))
  }

  for (stratum in strata) {
    stratum_label <- stratum$label
    df_stratum <- stratum$data

    for (outcome in outcomes) {
      outcome_var <- outcome[1]
      outcome_label <- outcome[2]
      if (!(outcome_var %in% names(df_stratum))) next

      for (comparison in comparison_categories) {
        df_model <- df_stratum[df_stratum$abuse %in% c(reference_category, comparison), , drop = FALSE]

        needed <- c(outcome_var, "age_year", "sex_male", "abuse")
        needed <- c(needed, add_covariates[add_covariates %in% names(df_model)])
        if (add_year_fe && year_col %in% names(df_model)) needed <- c(needed, year_col)
        if (!is.null(examiner_col) && examiner_col %in% names(df_model)) needed <- c(needed, examiner_col)
        if (!is.null(id_col) && id_col %in% names(df_model)) needed <- c(needed, id_col)
        needed <- unique(needed)

        df_model <- df_model[, needed, drop = FALSE]
        df_model <- df_model[stats::complete.cases(df_model[, setdiff(needed, id_col), drop = FALSE]), , drop = FALSE]
        if (nrow(df_model) < min_n) next

        df_model$comparison <- as.integer(df_model$abuse == comparison)

        extra_terms <- character(0)
        adjusted_for <- character(0)

        if (add_year_fe && year_col %in% names(df_model)) {
          extra_terms <- c(extra_terms, sprintf("factor(%s)", year_col))
          adjusted_for <- c(adjusted_for, "Year (FE)")
        }

        if (!is.null(examiner_col) && examiner_col %in% names(df_model)) {
          extra_terms <- c(extra_terms, sprintf("factor(%s)", examiner_col))
          adjusted_for <- c(adjusted_for, "Examiner (FE)")
        }

        for (cov in add_covariates) {
          if (!(cov %in% names(df_model))) next
          if (is.factor(df_model[[cov]]) || is.character(df_model[[cov]])) {
            extra_terms <- c(extra_terms, sprintf("factor(%s)", cov))
            adjusted_for <- c(adjusted_for, paste0(cov, " (FE)"))
          } else {
            extra_terms <- c(extra_terms, cov)
            adjusted_for <- c(adjusted_for, cov)
          }
        }

        adjusted_for_all <- c(ifelse(use_age_spline, "Age (spline)", "Age"), "Sex", adjusted_for)

        fit_res <- tryCatch(
          _fit_pairwise_logit(
            df_model,
            outcome_var,
            use_age_spline = use_age_spline,
            age_spline_df = age_spline_df,
            extra_terms = extra_terms,
            id_col = id_col,
            force_firth = force_firth
          ),
          error = function(e) NULL
        )

        if (!is.null(fit_res)) {
          beta <- fit_res$beta
          se <- fit_res$se
          p_val <- fit_res$p_value
          or_val <- exp(beta)
          ci_low <- ifelse(!is.na(se), exp(beta - 1.96 * se), NA_real_)
          ci_up <- ifelse(!is.na(se), exp(beta + 1.96 * se), NA_real_)

          results[[length(results) + 1]] <- list(
            Stratum = ifelse(is.null(stratify_by), "", stratum_label),
            Outcome = outcome_label,
            Comparison = paste0(comparison, " vs ", reference_category),
            N = nrow(df_model),
            Events = sum(df_model[[outcome_var]], na.rm = TRUE),
            `Odds Ratio` = sprintf("%.2f", or_val),
            `95% CI` = ifelse(!is.na(ci_low) && !is.na(ci_up), sprintf("(%.2f-%.2f)", ci_low, ci_up), "N/A"),
            `p-value` = format_p(p_val),
            Model = fit_res$model,
            Adjusted_for = paste(adjusted_for_all, collapse = ", ")
          )
        } else {
          results[[length(results) + 1]] <- list(
            Stratum = ifelse(is.null(stratify_by), "", stratum_label),
            Outcome = outcome_label,
            Comparison = paste0(comparison, " vs ", reference_category),
            N = nrow(df_model),
            Events = ifelse(outcome_var %in% names(df_model), sum(df_model[[outcome_var]], na.rm = TRUE), NA),
            `Odds Ratio` = "N/A",
            `95% CI` = "N/A",
            `p-value` = "N/A",
            Model = "N/A",
            Adjusted_for = paste(c("Age", "Sex", adjusted_for), collapse = ", ")
          )
        }
      }
    }
  }

  bind_rows_safe(results)
}

# -----------------------------------------------------------------------------
# Table 5.1: DMFT by dentition period and abuse type
# -----------------------------------------------------------------------------

create_table5_1_dmft_by_dentition_abuse <- function(df) {
  df_local <- df
  abuse_types <- get_levels(df_local$abuse)
  dentition_order <- c("primary_dentition", "mixed_dentition", "permanent_dentition")
  dentitions <- unique(as.character(df_local$dentition_type[!is.na(df_local$dentition_type)]))
  dentitions <- c(dentition_order[dentition_order %in% dentitions], setdiff(dentitions, dentition_order))
  results <- list()

  for (dentition in dentitions) {
    df_stage <- df_local[df_local$dentition_type == dentition, , drop = FALSE]
    groups <- lapply(abuse_types, function(abuse) safe_numeric(df_stage[df_stage$abuse == abuse, "DMFT_Index"]))
    groups <- lapply(groups, function(g) g[!is.na(g)])
    groups <- groups[vapply(groups, length, integer(1)) > 0]
    p_kw <- safe_kruskal_p(groups)
    p_val_str <- ifelse(is.na(p_kw), "N/A", format_p(p_kw))

    first_row <- TRUE
    for (abuse in abuse_types) {
      subset <- safe_numeric(df_stage[df_stage$abuse == abuse, "DMFT_Index"])
      subset <- subset[!is.na(subset)]
      if (length(subset) == 0) next
      q <- stats::quantile(subset, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      results[[length(results) + 1]] <- list(
        Dentition_Type = ifelse(first_row, dentition, ""),
        Abuse_Type = abuse,
        N = length(subset),
        Mean = sprintf("%.2f", mean(subset)),
        SD = ifelse(length(subset) > 1, sprintf("%.2f", stats::sd(subset)), "N/A"),
        Median = sprintf("%.1f", stats::median(subset)),
        `25%` = sprintf("%.1f", q[1]),
        `75%` = sprintf("%.1f", q[2]),
        Min = sprintf("%.0f", min(subset)),
        Max = sprintf("%.0f", max(subset)),
        `p-value (KW)` = ifelse(first_row, p_val_str, "")
      )
      first_row <- FALSE
    }
  }

  results[[length(results) + 1]] <- list(
    Dentition_Type = "=== OVERALL BY DENTITION PERIOD ===",
    Abuse_Type = "(Combined)",
    N = "---", Mean = "---", SD = "---", Median = "---",
    `25%` = "---", `75%` = "---", Min = "---", Max = "---", `p-value (KW)` = "---"
  )

  dentition_groups <- lapply(dentitions, function(d) safe_numeric(df_local[df_local$dentition_type == d, "DMFT_Index"]))
  dentition_groups <- lapply(dentition_groups, function(g) g[!is.na(g)])
  dentition_groups <- dentition_groups[vapply(dentition_groups, length, integer(1)) > 0]
  p_kw_dentition <- safe_kruskal_p(dentition_groups)
  p_val_dentition_str <- ifelse(is.na(p_kw_dentition), "N/A", format_p(p_kw_dentition))

  first_dentition <- TRUE
  for (dentition in dentitions) {
    subset <- safe_numeric(df_local[df_local$dentition_type == dentition, "DMFT_Index"])
    subset <- subset[!is.na(subset)]
    if (length(subset) > 0) {
      q <- stats::quantile(subset, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      results[[length(results) + 1]] <- list(
        Dentition_Type = dentition,
        Abuse_Type = "All abuse types",
        N = length(subset),
        Mean = sprintf("%.2f", mean(subset)),
        SD = ifelse(length(subset) > 1, sprintf("%.2f", stats::sd(subset)), "N/A"),
        Median = sprintf("%.1f", stats::median(subset)),
        `25%` = sprintf("%.1f", q[1]),
        `75%` = sprintf("%.1f", q[2]),
        Min = sprintf("%.0f", min(subset)),
        Max = sprintf("%.0f", max(subset)),
        `p-value (KW)` = ifelse(first_dentition, p_val_dentition_str, "")
      )
      first_dentition <- FALSE
    }
  }

  tidy_posthoc <- list()
  for (dentition in dentitions) {
    df_stage <- df_local[df_local$dentition_type == dentition, , drop = FALSE]
    groups <- lapply(abuse_types, function(abuse) safe_numeric(df_stage[df_stage$abuse == abuse, "DMFT_Index"]))
    groups <- lapply(groups, function(g) g[!is.na(g)])
    groups <- groups[vapply(groups, length, integer(1)) > 0]
    p_kw <- safe_kruskal_p(groups)
    if (!is.na(p_kw) && p_kw < 0.05) {
      dunn <- tryCatch(dunn_posthoc(df_stage, "DMFT_Index", "abuse", "bonferroni"), error = function(e) NULL)
      if (!is.null(dunn)) {
        for (pair in utils::combn(abuse_types, 2, simplify = FALSE)) {
          if (!(pair[1] %in% rownames(dunn$adjusted)) || !(pair[2] %in% colnames(dunn$adjusted))) next
          tidy_posthoc[[length(tidy_posthoc) + 1]] <- list(
            analysis_type = paste0("Table 5.1: ", dentition),
            variable = "DMFT_Index",
            group1 = pair[1],
            group2 = pair[2],
            p_unadjusted = as.numeric(dunn$unadjusted[pair[1], pair[2]]),
            p_adjusted = as.numeric(dunn$adjusted[pair[1], pair[2]]),
            significant = as.numeric(dunn$adjusted[pair[1], pair[2]]) < 0.05
          )
        }
      }
    }
  }

  if (length(dentition_groups) >= 2 && !is.na(p_kw_dentition) && p_kw_dentition < 0.05) {
    df_dent <- df_local[!is.na(df_local$dentition_type), , drop = FALSE]
    dunn <- tryCatch(dunn_posthoc(df_dent, "DMFT_Index", "dentition_type", "bonferroni"), error = function(e) NULL)
    if (!is.null(dunn)) {
      for (pair in utils::combn(dentitions, 2, simplify = FALSE)) {
        if (!(pair[1] %in% rownames(dunn$adjusted)) || !(pair[2] %in% colnames(dunn$adjusted))) next
        tidy_posthoc[[length(tidy_posthoc) + 1]] <- list(
          analysis_type = "Table 5.1: Dentition Period Overall",
          variable = "DMFT_Index",
          group1 = pair[1],
          group2 = pair[2],
          p_unadjusted = as.numeric(dunn$unadjusted[pair[1], pair[2]]),
          p_adjusted = as.numeric(dunn$adjusted[pair[1], pair[2]]),
          significant = as.numeric(dunn$adjusted[pair[1], pair[2]]) < 0.05
        )
      }
    }
  }

  results[[length(results) + 1]] <- list(
    Dentition_Type = "=== POST-HOC pairwise (Dunn's) ===",
    Abuse_Type = "(Only if KW p < 0.05)",
    N = "", Mean = "", SD = "", Median = "", `25%` = "", `75%` = "", Min = "", Max = "", `p-value (KW)` = ""
  )

  tidy_df <- bind_rows_safe(tidy_posthoc)
  if (nrow(tidy_df) > 0) {
    for (i in seq_len(nrow(tidy_df))) {
      tp <- tidy_df[i, ]
      results[[length(results) + 1]] <- list(
        Dentition_Type = paste0("Post-hoc: ", tp$analysis_type),
        Abuse_Type = paste0(tp$group1, " vs ", tp$group2),
        N = ifelse(tp$significant, "Sig", "n.s."),
        Mean = sprintf("padj=%.4f", tp$p_adjusted),
        SD = sprintf("pun=%.4f", tp$p_unadjusted),
        Median = "", `25%` = "", `75%` = "", Min = "", Max = "", `p-value (KW)` = ""
      )
    }
  }

  list(summary = bind_rows_safe(results), tidy_posthoc = tidy_df)
}

# -----------------------------------------------------------------------------
# Table 5: DMFT by life stage and abuse type
# -----------------------------------------------------------------------------

create_table5_dmft_by_lifestage_abuse <- function(df) {
  df_local <- df
  abuse_types <- get_levels(df_local$abuse)
  life_stage_order <- c("Early Childhood (2-6)", "Middle Childhood (7-12)", "Adolescence (13-18)")
  life_stages <- unique(as.character(df_local$age_group[!is.na(df_local$age_group)]))
  life_stages <- c(life_stage_order[life_stage_order %in% life_stages], setdiff(life_stages, life_stage_order))
  results <- list()

  for (life_stage in life_stages) {
    df_stage <- df_local[df_local$age_group == life_stage, , drop = FALSE]
    groups <- lapply(abuse_types, function(abuse) safe_numeric(df_stage[df_stage$abuse == abuse, "DMFT_Index"]))
    groups <- lapply(groups, function(g) g[!is.na(g)])
    groups <- groups[vapply(groups, length, integer(1)) > 0]
    p_kw <- safe_kruskal_p(groups)
    p_val_str <- ifelse(is.na(p_kw), "N/A", format_p(p_kw))

    first_row <- TRUE
    for (abuse in abuse_types) {
      subset <- safe_numeric(df_stage[df_stage$abuse == abuse, "DMFT_Index"])
      subset <- subset[!is.na(subset)]
      if (length(subset) == 0) next
      q <- stats::quantile(subset, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      results[[length(results) + 1]] <- list(
        Life_Stage = ifelse(first_row, life_stage, ""),
        Abuse_Type = abuse,
        N = length(subset),
        Mean = sprintf("%.2f", mean(subset)),
        SD = sprintf("%.2f", safe_sd(subset)),
        Median = sprintf("%.1f", stats::median(subset)),
        `25%` = sprintf("%.1f", q[1]),
        `75%` = sprintf("%.1f", q[2]),
        Min = sprintf("%.0f", min(subset)),
        Max = sprintf("%.0f", max(subset)),
        `p-value (KW)` = ifelse(first_row, p_val_str, "")
      )
      first_row <- FALSE
    }
  }

  results[[length(results) + 1]] <- list(
    Life_Stage = "=== OVERALL BY LIFE STAGE ===",
    Abuse_Type = "(Combined)",
    N = "---", Mean = "---", SD = "---", Median = "---",
    `25%` = "---", `75%` = "---", Min = "---", Max = "---", `p-value (KW)` = "---"
  )

  life_stage_groups <- lapply(life_stages, function(ls) safe_numeric(df_local[df_local$age_group == ls, "DMFT_Index"]))
  life_stage_groups <- lapply(life_stage_groups, function(g) g[!is.na(g)])
  life_stage_groups <- life_stage_groups[vapply(life_stage_groups, length, integer(1)) > 0]
  p_kw_lifestage <- safe_kruskal_p(life_stage_groups)
  p_val_lifestage_str <- ifelse(is.na(p_kw_lifestage), "N/A", format_p(p_kw_lifestage))

  first_lifestage <- TRUE
  for (life_stage in life_stages) {
    subset <- safe_numeric(df_local[df_local$age_group == life_stage, "DMFT_Index"])
    subset <- subset[!is.na(subset)]
    if (length(subset) > 0) {
      q <- stats::quantile(subset, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      results[[length(results) + 1]] <- list(
        Life_Stage = life_stage,
        Abuse_Type = "All abuse types",
        N = length(subset),
        Mean = sprintf("%.2f", mean(subset)),
        SD = sprintf("%.2f", safe_sd(subset)),
        Median = sprintf("%.1f", stats::median(subset)),
        `25%` = sprintf("%.1f", q[1]),
        `75%` = sprintf("%.1f", q[2]),
        Min = sprintf("%.0f", min(subset)),
        Max = sprintf("%.0f", max(subset)),
        `p-value (KW)` = ifelse(first_lifestage, p_val_lifestage_str, "")
      )
      first_lifestage <- FALSE
    }
  }

  tidy_posthoc <- list()
  for (life_stage in life_stages) {
    df_stage <- df_local[df_local$age_group == life_stage, , drop = FALSE]
    groups <- lapply(abuse_types, function(abuse) safe_numeric(df_stage[df_stage$abuse == abuse, "DMFT_Index"]))
    groups <- lapply(groups, function(g) g[!is.na(g)])
    groups <- groups[vapply(groups, length, integer(1)) > 0]
    p_kw <- safe_kruskal_p(groups)
    if (!is.na(p_kw) && p_kw < 0.05) {
      dunn <- tryCatch(dunn_posthoc(df_stage, "DMFT_Index", "abuse", "bonferroni"), error = function(e) NULL)
      if (!is.null(dunn)) {
        for (pair in utils::combn(abuse_types, 2, simplify = FALSE)) {
          if (!(pair[1] %in% rownames(dunn$adjusted)) || !(pair[2] %in% colnames(dunn$adjusted))) next
          tidy_posthoc[[length(tidy_posthoc) + 1]] <- list(
            analysis_type = paste0("Table 5: ", life_stage),
            variable = "DMFT_Index",
            group1 = pair[1],
            group2 = pair[2],
            p_unadjusted = as.numeric(dunn$unadjusted[pair[1], pair[2]]),
            p_adjusted = as.numeric(dunn$adjusted[pair[1], pair[2]]),
            significant = as.numeric(dunn$adjusted[pair[1], pair[2]]) < 0.05
          )
        }
      }
    }
  }

  if (length(life_stage_groups) >= 2 && !is.na(p_kw_lifestage) && p_kw_lifestage < 0.05) {
    df_ls <- df_local[!is.na(df_local$age_group), , drop = FALSE]
    dunn <- tryCatch(dunn_posthoc(df_ls, "DMFT_Index", "age_group", "bonferroni"), error = function(e) NULL)
    if (!is.null(dunn)) {
      for (pair in utils::combn(life_stages, 2, simplify = FALSE)) {
        if (!(pair[1] %in% rownames(dunn$adjusted)) || !(pair[2] %in% colnames(dunn$adjusted))) next
        tidy_posthoc[[length(tidy_posthoc) + 1]] <- list(
          analysis_type = "Table 5: Life Stage Overall",
          variable = "DMFT_Index",
          group1 = pair[1],
          group2 = pair[2],
          p_unadjusted = as.numeric(dunn$unadjusted[pair[1], pair[2]]),
          p_adjusted = as.numeric(dunn$adjusted[pair[1], pair[2]]),
          significant = as.numeric(dunn$adjusted[pair[1], pair[2]]) < 0.05
        )
      }
    }
  }

  results[[length(results) + 1]] <- list(
    Life_Stage = "=== POST-HOC pairwise (Dunn's) ===",
    Abuse_Type = "(Only if KW p < 0.05)",
    N = "", Mean = "", SD = "", Median = "", `25%` = "", `75%` = "", Min = "", Max = "", `p-value (KW)` = ""
  )

  tidy_df <- bind_rows_safe(tidy_posthoc)
  if (nrow(tidy_df) > 0) {
    for (i in seq_len(nrow(tidy_df))) {
      tp <- tidy_df[i, ]
      results[[length(results) + 1]] <- list(
        Life_Stage = paste0("Post-hoc: ", tp$analysis_type),
        Abuse_Type = paste0(tp$group1, " vs ", tp$group2),
        N = ifelse(tp$significant, "Sig", "n.s."),
        Mean = sprintf("padj=%.4f", tp$p_adjusted),
        SD = sprintf("pun=%.4f", tp$p_unadjusted),
        Median = "", `25%` = "", `75%` = "", Min = "", Max = "", `p-value (KW)` = ""
      )
    }
  }

  list(summary = bind_rows_safe(results), tidy_posthoc = tidy_df)
}

# -----------------------------------------------------------------------------
# Table 5.5: caries prevalence and treatment status
# -----------------------------------------------------------------------------

create_table5_5_caries_prevalence_treatment <- function(df) {
  df_local <- df
  abuse_types <- get_levels(df_local$abuse)
  results <- list()

  header <- list(Variable = "=== CARIES PREVALENCE ===", Category = "")
  for (a in abuse_types) header[[a]] <- ""
  header[["Total"]] <- ""
  header[["p-value"]] <- ""
  results[[length(results) + 1]] <- header

  if (!("filled_total" %in% names(df_local)) && all(c("Perm_F", "Baby_f") %in% names(df_local))) {
    df_local$filled_total <- add_fill0(df_local$Perm_F, df_local$Baby_f)
  }
  if (!("decayed_total" %in% names(df_local)) && all(c("Perm_D", "Baby_d") %in% names(df_local))) {
    df_local$decayed_total <- add_fill0(df_local$Perm_D, df_local$Baby_d)
  }
  if (!("missing_total" %in% names(df_local)) && all(c("Perm_M", "Baby_m") %in% names(df_local))) {
    df_local$missing_total <- add_fill0(df_local$Perm_M, df_local$Baby_m)
  }

  prevalence_vars <- list(
    c("Children with Caries", "DMFT_Index", "DMFT_Index > 0"),
    c("Untreated Caries (Decayed)", "decayed_total", "decayed_total > 0"),
    c("Missing Teeth (Missing)", "missing_total", "missing_total > 0"),
    c("Filled Teeth (Filled)", "filled_total", "filled_total > 0")
  )

  for (var_info in prevalence_vars) {
    var_label <- var_info[1]
    var_col <- var_info[2]
    var_cat <- var_info[3]
    if (!(var_col %in% names(df_local))) next

    row <- list(Variable = var_label, Category = var_cat)
    for (abuse in abuse_types) {
      subset <- df_local[df_local$abuse == abuse, , drop = FALSE]
      n_total <- nrow(subset)
      n_prev <- sum(subset[[var_col]] > 0, na.rm = TRUE)
      pct <- ifelse(n_total > 0, n_prev / n_total * 100, 0)
      row[[abuse]] <- sprintf("%d/%d (%.1f%%)", n_prev, n_total, pct)
    }
    n_total_all <- nrow(df_local)
    n_prev_all <- sum(df_local[[var_col]] > 0, na.rm = TRUE)
    pct_all <- ifelse(n_total_all > 0, n_prev_all / n_total_all * 100, 0)
    row[["Total"]] <- sprintf("%d/%d (%.1f%%)", n_prev_all, n_total_all, pct_all)

    temp_col <- paste0("has_", var_col)
    df_local[[temp_col]] <- as.integer(!is.na(df_local[[var_col]]) & df_local[[var_col]] > 0)
    p_val <- safe_chisq_p(df_local, "abuse", temp_col)
    row[["p-value"]] <- ifelse(is.na(p_val), "N/A", format_p(p_val))
    results[[length(results) + 1]] <- row
  }

  header <- list(Variable = "=== TREATMENT STATUS ===", Category = "")
  for (a in abuse_types) header[[a]] <- ""
  header[["Total"]] <- ""
  header[["p-value"]] <- ""
  results[[length(results) + 1]] <- header

  df_local$filled_total <- add_fill0(df_local$Perm_F, df_local$Baby_f)
  df_with_caries <- df_local[df_local$DMFT_Index > 0 & !is.na(df_local$DMFT_Index), , drop = FALSE]

  row_fully_treated <- list(Variable = "Fully Treated Caries", Category = "f+F = DMFT_Index")
  for (abuse in abuse_types) {
    subset <- df_with_caries[df_with_caries$abuse == abuse, , drop = FALSE]
    n_total <- nrow(subset)
    n_fully <- sum(subset$filled_total == subset$DMFT_Index, na.rm = TRUE)
    pct <- ifelse(n_total > 0, n_fully / n_total * 100, 0)
    row_fully_treated[[abuse]] <- sprintf("%d/%d (%.1f%%)", n_fully, n_total, pct)
  }
  n_total_caries <- nrow(df_with_caries)
  n_fully_all <- sum(df_with_caries$filled_total == df_with_caries$DMFT_Index, na.rm = TRUE)
  pct_fully <- ifelse(n_total_caries > 0, n_fully_all / n_total_caries * 100, 0)
  row_fully_treated[["Total"]] <- sprintf("%d/%d (%.1f%%)", n_fully_all, n_total_caries, pct_fully)
  df_with_caries$is_fully_treated <- as.integer(df_with_caries$filled_total == df_with_caries$DMFT_Index)
  p_val <- safe_chisq_p(df_with_caries, "abuse", "is_fully_treated")
  row_fully_treated[["p-value"]] <- ifelse(is.na(p_val), "N/A", format_p(p_val))
  results[[length(results) + 1]] <- row_fully_treated

  row_no_filled <- list(Variable = "No Filled Teeth", Category = "f+F = 0 (Among Caries Active)")
  for (abuse in abuse_types) {
    subset <- df_with_caries[df_with_caries$abuse == abuse, , drop = FALSE]
    n_total <- nrow(subset)
    n_no_filled <- sum(subset$filled_total == 0, na.rm = TRUE)
    pct <- ifelse(n_total > 0, n_no_filled / n_total * 100, 0)
    row_no_filled[[abuse]] <- sprintf("%d/%d (%.1f%%)", n_no_filled, n_total, pct)
  }
  n_no_filled_all <- sum(df_with_caries$filled_total == 0, na.rm = TRUE)
  pct_no_filled <- ifelse(n_total_caries > 0, n_no_filled_all / n_total_caries * 100, 0)
  row_no_filled[["Total"]] <- sprintf("%d/%d (%.1f%%)", n_no_filled_all, n_total_caries, pct_no_filled)
  df_local$has_no_filled <- as.integer(df_local$DMFT_Index > 0 & df_local$filled_total == 0)
  p_val <- safe_chisq_p(df_local, "abuse", "has_no_filled")
  row_no_filled[["p-value"]] <- ifelse(is.na(p_val), "N/A", format_p(p_val))
  results[[length(results) + 1]] <- row_no_filled

  header <- list(Variable = "=== DMFT WITH C0 ===", Category = "")
  for (a in abuse_types) header[[a]] <- ""
  header[["Total"]] <- ""
  header[["p-value"]] <- ""
  results[[length(results) + 1]] <- header

  c0_vars <- list(
    c("DMFT_C0", "Total DMFT + C0"),
    c("Perm_DMFT_C0", "Permanent DMFT + C0"),
    c("Baby_DMFT_C0", "Primary dmft + C0")
  )

  for (var_info in c0_vars) {
    var_name <- var_info[1]
    var_label <- var_info[2]
    if (!(var_name %in% names(df_local))) next

    row_mean <- list(Variable = var_label, Category = "Mean ± SD")
    for (abuse in abuse_types) {
      row_mean[[abuse]] <- format_mean_sd(df_local[df_local$abuse == abuse, var_name], digits = 2)
    }
    row_mean[["Total"]] <- format_mean_sd(df_local[[var_name]], digits = 2)
    groups <- lapply(abuse_types, function(abuse) safe_numeric(df_local[df_local$abuse == abuse, var_name]))
    groups <- lapply(groups, function(g) g[!is.na(g)])
    p_val <- safe_kruskal_p(groups)
    row_mean[["p-value"]] <- ifelse(is.na(p_val), "N/A", format_p(p_val))
    results[[length(results) + 1]] <- row_mean

    row_median <- list(Variable = "", Category = "Median [IQR]")
    for (abuse in abuse_types) {
      row_median[[abuse]] <- format_median_iqr(df_local[df_local$abuse == abuse, var_name], digits = 1)
    }
    row_median[["Total"]] <- format_median_iqr(df_local[[var_name]], digits = 1)
    row_median[["p-value"]] <- ""
    results[[length(results) + 1]] <- row_median
  }

  header <- list(Variable = "=== CARIES PREVALENCE (INCL. C0) ===", Category = "")
  for (a in abuse_types) header[[a]] <- ""
  header[["Total"]] <- ""
  header[["p-value"]] <- ""
  results[[length(results) + 1]] <- header

  if ("DMFT_C0" %in% names(df_local)) {
    row_c0 <- list(Variable = "Children with Caries (incl. C0)", Category = "DMFT_C0 > 0")
    for (abuse in abuse_types) {
      subset <- df_local[df_local$abuse == abuse, , drop = FALSE]
      n_total <- nrow(subset)
      n_caries <- sum(subset$DMFT_C0 > 0, na.rm = TRUE)
      pct <- ifelse(n_total > 0, n_caries / n_total * 100, 0)
      row_c0[[abuse]] <- sprintf("%d/%d (%.1f%%)", n_caries, n_total, pct)
    }
    n_all <- nrow(df_local)
    n_caries_all <- sum(df_local$DMFT_C0 > 0, na.rm = TRUE)
    row_c0[["Total"]] <- sprintf("%d/%d (%.1f%%)", n_caries_all, n_all, ifelse(n_all > 0, n_caries_all / n_all * 100, 0))
    df_local$has_caries_c0 <- as.integer(df_local$DMFT_C0 > 0)
    p_val <- safe_chisq_p(df_local, "abuse", "has_caries_c0")
    row_c0[["p-value"]] <- ifelse(is.na(p_val), "N/A", format_p(p_val))
    results[[length(results) + 1]] <- row_c0
  }

  header <- list(Variable = "=== POST-HOC (C0) ===", Category = "")
  for (a in abuse_types) header[[a]] <- ""
  header[["Total"]] <- ""
  header[["p-value"]] <- ""
  results[[length(results) + 1]] <- header

  list(summary = bind_rows_safe(results), tidy_posthoc = data.frame())
}

# -----------------------------------------------------------------------------
# Table 6: DMFT by dentition type and abuse type
# -----------------------------------------------------------------------------

create_table6_dmft_by_dentition_abuse <- function(df) {
  required_cols <- c("DMFT_Index", "Present_Teeth", "Present_Perm_Teeth", "abuse")
  for (col in required_cols) {
    if (!(col %in% names(df))) {
      message("   ⚠ '", col, "' column not found")
      return(list(
        summary_table = data.frame(),
        within_dentition_posthoc = data.frame(),
        within_abuse_posthoc = data.frame(),
        overall_dentition_posthoc = data.frame()
      ))
    }
  }

  df_analysis <- df
  if (!("dentition_type" %in% names(df_analysis))) {
    if (!("Present_Baby_Teeth" %in% names(df_analysis))) {
      message("   ⚠ 'Present_Baby_Teeth' column not found")
      return(list(
        summary_table = data.frame(),
        within_dentition_posthoc = data.frame(),
        within_abuse_posthoc = data.frame(),
        overall_dentition_posthoc = data.frame()
      ))
    }
  }

  abuse_types <- get_levels(df_analysis$abuse)
  dentition_order <- c("primary_dentition", "mixed_dentition", "permanent_dentition")
  df_analysis <- df_analysis[df_analysis$dentition_type %in% dentition_order, , drop = FALSE]
  summary_results <- list()

  for (dent_type in dentition_order) {
    df_dent <- df_analysis[df_analysis$dentition_type == dent_type, , drop = FALSE]
    if (nrow(df_dent) == 0) next

    groups <- lapply(abuse_types, function(abuse) safe_numeric(df_dent[df_dent$abuse == abuse, "DMFT_Index"]))
    groups <- lapply(groups, function(g) g[!is.na(g)])
    groups <- groups[vapply(groups, length, integer(1)) > 0]
    p_kw <- safe_kruskal_p(groups)
    p_val_str <- ifelse(is.na(p_kw), "N/A", format_p(p_kw))

    overall_subset <- safe_numeric(df_dent$DMFT_Index)
    overall_subset <- overall_subset[!is.na(overall_subset)]
    if (length(overall_subset) > 0) {
      q <- stats::quantile(overall_subset, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      sd_val <- safe_sd(overall_subset)
      summary_results[[length(summary_results) + 1]] <- list(
        Dentition_Type = dent_type,
        Abuse_Type = "Total",
        N = length(overall_subset),
        Mean = round(mean(overall_subset), 2),
        SD = round(sd_val, 2),
        Median = round(stats::median(overall_subset), 2),
        IQR = sprintf("%.2f-%.2f", q[1], q[2]),
        Min = round(min(overall_subset), 2),
        Max = round(max(overall_subset), 2),
        Mean_SD = ifelse(is.na(sd_val), sprintf("%.2f", mean(overall_subset)), sprintf("%.2f ± %.2f", mean(overall_subset), sd_val)),
        Median_IQR = sprintf("%.1f [%.1f-%.1f]", stats::median(overall_subset), q[1], q[2]),
        `Min-Max` = sprintf("%.1f-%.1f", min(overall_subset), max(overall_subset)),
        `p-value (KW within dentition)` = p_val_str
      )
    }

    first_row <- FALSE
    for (abuse in abuse_types) {
      subset <- safe_numeric(df_dent[df_dent$abuse == abuse, "DMFT_Index"])
      subset <- subset[!is.na(subset)]
      if (length(subset) == 0) next
      q <- stats::quantile(subset, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      sd_val <- safe_sd(subset)
      summary_results[[length(summary_results) + 1]] <- list(
        Dentition_Type = ifelse(first_row, dent_type, ""),
        Abuse_Type = abuse,
        N = length(subset),
        Mean = round(mean(subset), 2),
        SD = round(sd_val, 2),
        Median = round(stats::median(subset), 2),
        IQR = sprintf("%.2f-%.2f", q[1], q[2]),
        Min = round(min(subset), 2),
        Max = round(max(subset), 2),
        Mean_SD = ifelse(is.na(sd_val), sprintf("%.2f", mean(subset)), sprintf("%.2f ± %.2f", mean(subset), sd_val)),
        Median_IQR = sprintf("%.1f [%.1f-%.1f]", stats::median(subset), q[1], q[2]),
        `Min-Max` = sprintf("%.1f-%.1f", min(subset), max(subset)),
        `p-value (KW within dentition)` = ifelse(first_row, p_val_str, "")
      )
      first_row <- FALSE
    }
  }

  within_dentition_posthoc <- list()
  for (dent_type in dentition_order) {
    df_dent <- df_analysis[df_analysis$dentition_type == dent_type & !is.na(df_analysis$DMFT_Index) & !is.na(df_analysis$abuse), , drop = FALSE]
    if (nrow(df_dent) == 0) next
    groups <- lapply(abuse_types, function(abuse) safe_numeric(df_dent[df_dent$abuse == abuse, "DMFT_Index"]))
    groups <- lapply(groups, function(g) g[!is.na(g)])
    groups <- groups[vapply(groups, length, integer(1)) > 0]
    if (length(groups) < 2) next
    p_kw <- safe_kruskal_p(groups)
    if (is.na(p_kw) || p_kw >= 0.05) next

    df_dent$`_rank` <- rank(df_dent$DMFT_Index, ties.method = "average")
    mean_ranks <- tapply(df_dent$`_rank`, df_dent$abuse, mean)
    dunn <- tryCatch(dunn_posthoc(df_dent, "DMFT_Index", "abuse", "bonferroni"), error = function(e) NULL)
    if (is.null(dunn)) next

    for (pair in utils::combn(abuse_types, 2, simplify = FALSE)) {
      abuse1 <- pair[1]
      abuse2 <- pair[2]
      if (!(abuse1 %in% rownames(dunn$adjusted)) || !(abuse2 %in% colnames(dunn$adjusted))) next
      vals1 <- safe_numeric(df_dent[df_dent$abuse == abuse1, "DMFT_Index"])
      vals2 <- safe_numeric(df_dent[df_dent$abuse == abuse2, "DMFT_Index"])
      vals1 <- vals1[!is.na(vals1)]
      vals2 <- vals2[!is.na(vals2)]
      if (length(vals1) == 0 || length(vals2) == 0) next
      q1 <- stats::quantile(vals1, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      q2 <- stats::quantile(vals2, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      sd1 <- safe_sd(vals1)
      sd2 <- safe_sd(vals2)
      p_adj <- as.numeric(dunn$adjusted[abuse1, abuse2])
      p_unadj <- as.numeric(dunn$unadjusted[abuse1, abuse2])
      within_dentition_posthoc[[length(within_dentition_posthoc) + 1]] <- list(
        Analysis = "Within dentition: abuse subtype comparison",
        Dentition_Type = dent_type,
        Variable = "DMFT_Index",
        Group1 = abuse1,
        Group2 = abuse2,
        Comparison = paste0(abuse1, " vs ", abuse2),
        Group1_n = length(vals1),
        Group2_n = length(vals2),
        Group1_Mean = round(mean(vals1), 2),
        Group2_Mean = round(mean(vals2), 2),
        Group1_SD = round(sd1, 2),
        Group2_SD = round(sd2, 2),
        Group1_Median = round(stats::median(vals1), 2),
        Group2_Median = round(stats::median(vals2), 2),
        Group1_IQR = sprintf("%.2f-%.2f", q1[1], q1[2]),
        Group2_IQR = sprintf("%.2f-%.2f", q2[1], q2[2]),
        Group1_Mean_SD = ifelse(is.na(sd1), sprintf("%.2f", mean(vals1)), sprintf("%.2f ± %.2f", mean(vals1), sd1)),
        Group2_Mean_SD = ifelse(is.na(sd2), sprintf("%.2f", mean(vals2)), sprintf("%.2f ± %.2f", mean(vals2), sd2)),
        Group1_Median_IQR = sprintf("%.2f [%.2f-%.2f]", stats::median(vals1), q1[1], q1[2]),
        Group2_Median_IQR = sprintf("%.2f [%.2f-%.2f]", stats::median(vals2), q2[1], q2[2]),
        Group1_Mean_Rank = round(mean_ranks[abuse1], 2),
        Group2_Mean_Rank = round(mean_ranks[abuse2], 2),
        KW_p_value = format_p(p_kw),
        `p-value (unadjusted)` = format_p(p_unadj),
        `p-value (adjusted)` = format_p(p_adj),
        Significant = ifelse(p_adj < 0.05, "Yes", "No")
      )
    }
  }

  within_abuse_posthoc <- list()
  for (abuse in abuse_types) {
    df_abuse <- df_analysis[df_analysis$abuse == abuse & !is.na(df_analysis$DMFT_Index) & !is.na(df_analysis$dentition_type), , drop = FALSE]
    if (nrow(df_abuse) == 0) next
    groups <- lapply(dentition_order, function(dent) safe_numeric(df_abuse[df_abuse$dentition_type == dent, "DMFT_Index"]))
    groups <- lapply(groups, function(g) g[!is.na(g)])
    groups <- groups[vapply(groups, length, integer(1)) > 0]
    if (length(groups) < 2) next
    p_kw <- safe_kruskal_p(groups)
    if (is.na(p_kw) || p_kw >= 0.05) next

    df_abuse$`_rank` <- rank(df_abuse$DMFT_Index, ties.method = "average")
    mean_ranks <- tapply(df_abuse$`_rank`, df_abuse$dentition_type, mean)
    dunn <- tryCatch(dunn_posthoc(df_abuse, "DMFT_Index", "dentition_type", "bonferroni"), error = function(e) NULL)
    if (is.null(dunn)) next

    for (pair in utils::combn(dentition_order, 2, simplify = FALSE)) {
      dent1 <- pair[1]
      dent2 <- pair[2]
      if (!(dent1 %in% rownames(dunn$adjusted)) || !(dent2 %in% colnames(dunn$adjusted))) next
      vals1 <- safe_numeric(df_abuse[df_abuse$dentition_type == dent1, "DMFT_Index"])
      vals2 <- safe_numeric(df_abuse[df_abuse$dentition_type == dent2, "DMFT_Index"])
      vals1 <- vals1[!is.na(vals1)]
      vals2 <- vals2[!is.na(vals2)]
      if (length(vals1) == 0 || length(vals2) == 0) next
      q1 <- stats::quantile(vals1, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      q2 <- stats::quantile(vals2, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
      sd1 <- safe_sd(vals1)
      sd2 <- safe_sd(vals2)
      p_adj <- as.numeric(dunn$adjusted[dent1, dent2])
      p_unadj <- as.numeric(dunn$unadjusted[dent1, dent2])
      within_abuse_posthoc[[length(within_abuse_posthoc) + 1]] <- list(
        Analysis = "Within abuse subtype: dentition comparison",
        Abuse_Type = abuse,
        Variable = "DMFT_Index",
        Group1 = dent1,
        Group2 = dent2,
        Comparison = paste0(dent1, " vs ", dent2),
        Group1_n = length(vals1),
        Group2_n = length(vals2),
        Group1_Mean = round(mean(vals1), 2),
        Group2_Mean = round(mean(vals2), 2),
        Group1_SD = round(sd1, 2),
        Group2_SD = round(sd2, 2),
        Group1_Median = round(stats::median(vals1), 2),
        Group2_Median = round(stats::median(vals2), 2),
        Group1_IQR = sprintf("%.2f-%.2f", q1[1], q1[2]),
        Group2_IQR = sprintf("%.2f-%.2f", q2[1], q2[2]),
        Group1_Mean_SD = ifelse(is.na(sd1), sprintf("%.2f", mean(vals1)), sprintf("%.2f ± %.2f", mean(vals1), sd1)),
        Group2_Mean_SD = ifelse(is.na(sd2), sprintf("%.2f", mean(vals2)), sprintf("%.2f ± %.2f", mean(vals2), sd2)),
        Group1_Median_IQR = sprintf("%.2f [%.2f-%.2f]", stats::median(vals1), q1[1], q1[2]),
        Group2_Median_IQR = sprintf("%.2f [%.2f-%.2f]", stats::median(vals2), q2[1], q2[2]),
        Group1_Mean_Rank = round(mean_ranks[dent1], 2),
        Group2_Mean_Rank = round(mean_ranks[dent2], 2),
        KW_p_value = format_p(p_kw),
        `p-value (unadjusted)` = format_p(p_unadj),
        `p-value (adjusted)` = format_p(p_adj),
        Significant = ifelse(p_adj < 0.05, "Yes", "No")
      )
    }
  }

  overall_dentition_posthoc <- list()
  df_overall <- df_analysis[!is.na(df_analysis$DMFT_Index) & !is.na(df_analysis$dentition_type), , drop = FALSE]
  groups <- lapply(dentition_order, function(dent) safe_numeric(df_overall[df_overall$dentition_type == dent, "DMFT_Index"]))
  groups <- lapply(groups, function(g) g[!is.na(g)])
  groups <- groups[vapply(groups, length, integer(1)) > 0]

  if (length(groups) >= 2) {
    p_kw <- safe_kruskal_p(groups)
    if (!is.na(p_kw) && p_kw < 0.05) {
      df_overall$`_rank` <- rank(df_overall$DMFT_Index, ties.method = "average")
      mean_ranks <- tapply(df_overall$`_rank`, df_overall$dentition_type, mean)
      dunn <- tryCatch(dunn_posthoc(df_overall, "DMFT_Index", "dentition_type", "bonferroni"), error = function(e) NULL)
      if (!is.null(dunn)) {
        for (pair in utils::combn(dentition_order, 2, simplify = FALSE)) {
          dent1 <- pair[1]
          dent2 <- pair[2]
          if (!(dent1 %in% rownames(dunn$adjusted)) || !(dent2 %in% colnames(dunn$adjusted))) next
          vals1 <- safe_numeric(df_overall[df_overall$dentition_type == dent1, "DMFT_Index"])
          vals2 <- safe_numeric(df_overall[df_overall$dentition_type == dent2, "DMFT_Index"])
          vals1 <- vals1[!is.na(vals1)]
          vals2 <- vals2[!is.na(vals2)]
          if (length(vals1) == 0 || length(vals2) == 0) next
          q1 <- stats::quantile(vals1, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
          q2 <- stats::quantile(vals2, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
          sd1 <- safe_sd(vals1)
          sd2 <- safe_sd(vals2)
          p_adj <- as.numeric(dunn$adjusted[dent1, dent2])
          p_unadj <- as.numeric(dunn$unadjusted[dent1, dent2])
          overall_dentition_posthoc[[length(overall_dentition_posthoc) + 1]] <- list(
            Analysis = "Overall dentition comparison",
            Variable = "DMFT_Index",
            Group1 = dent1,
            Group2 = dent2,
            Comparison = paste0(dent1, " vs ", dent2),
            Group1_n = length(vals1),
            Group2_n = length(vals2),
            Group1_Mean = round(mean(vals1), 2),
            Group2_Mean = round(mean(vals2), 2),
            Group1_SD = round(sd1, 2),
            Group2_SD = round(sd2, 2),
            Group1_Median = round(stats::median(vals1), 2),
            Group2_Median = round(stats::median(vals2), 2),
            Group1_IQR = sprintf("%.2f-%.2f", q1[1], q1[2]),
            Group2_IQR = sprintf("%.2f-%.2f", q2[1], q2[2]),
            Group1_Mean_SD = ifelse(is.na(sd1), sprintf("%.2f", mean(vals1)), sprintf("%.2f ± %.2f", mean(vals1), sd1)),
            Group2_Mean_SD = ifelse(is.na(sd2), sprintf("%.2f", mean(vals2)), sprintf("%.2f ± %.2f", mean(vals2), sd2)),
            Group1_Median_IQR = sprintf("%.2f [%.2f-%.2f]", stats::median(vals1), q1[1], q1[2]),
            Group2_Median_IQR = sprintf("%.2f [%.2f-%.2f]", stats::median(vals2), q2[1], q2[2]),
            Group1_Mean_Rank = round(mean_ranks[dent1], 2),
            Group2_Mean_Rank = round(mean_ranks[dent2], 2),
            KW_p_value = format_p(p_kw),
            `p-value (unadjusted)` = format_p(p_unadj),
            `p-value (adjusted)` = format_p(p_adj),
            Significant = ifelse(p_adj < 0.05, "Yes", "No")
          )
        }
      }
    }
  }

  list(
    summary_table = bind_rows_safe(summary_results),
    within_dentition_posthoc = bind_rows_safe(within_dentition_posthoc),
    within_abuse_posthoc = bind_rows_safe(within_abuse_posthoc),
    overall_dentition_posthoc = bind_rows_safe(overall_dentition_posthoc)
  )
}

# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

p_to_star <- function(p_val) {
  p <- p_value_to_numeric(p_val)
  if (is.na(p)) return("")
  if (p < 0.001) return("***")
  if (p < 0.01) return("**")
  if (p < 0.05) return("*")
  ""
}

plot_overall_dentition_refined <- function(
  df,
  posthoc_df,
  y_col = "DMFT_Index",
  xlabel = NULL,
  ylabel = NULL,
  title = NULL,
  title_fontsize = 14,
  label_fontsize = 14,
  tick_fontsize = 12,
  save_path = NULL
) {
  dentition_order <- c("primary_dentition", "mixed_dentition", "permanent_dentition")
  df_plot <- df[df$dentition_type %in% dentition_order, , drop = FALSE]
  if (nrow(df_plot) == 0 || !(y_col %in% names(df_plot))) return(invisible(NULL))
  df_plot$dentition_type <- factor(df_plot$dentition_type, levels = dentition_order)

  n_df <- df_plot %>% dplyr::group_by(dentition_type) %>% dplyr::summarise(n = sum(!is.na(.data[[y_col]])), .groups = "drop")
  label_map <- setNames(paste0(gsub("_", " ", tools::toTitleCase(as.character(n_df$dentition_type))), "\n(n=", n_df$n, ")"), as.character(n_df$dentition_type))

  p <- ggplot2::ggplot(df_plot, ggplot2::aes(x = dentition_type, y = .data[[y_col]])) +
    ggplot2::geom_boxplot(outlier.shape = NA) +
    ggplot2::geom_jitter(width = 0.15, alpha = 0.4, size = 1.5) +
    ggplot2::stat_summary(fun = mean, geom = "point", shape = 18, size = 3) +
    ggplot2::stat_summary(fun = mean, geom = "text", ggplot2::aes(label = sprintf("%.2f", after_stat(y))), vjust = -0.5, fontface = "bold") +
    ggplot2::scale_x_discrete(labels = label_map) +
    ggplot2::labs(x = xlabel, y = ifelse(is.null(ylabel), y_col, ylabel), title = title) +
    ggplot2::theme_bw(base_size = tick_fontsize) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(face = "bold"), axis.title = ggplot2::element_text(face = "bold"))

  if (!is.null(posthoc_df) && nrow(posthoc_df) > 0 && "Significant" %in% names(posthoc_df)) {
    sig_results <- posthoc_df[posthoc_df$Significant == "Yes", , drop = FALSE]
    if (nrow(sig_results) > 0) {
      y_max <- max(df_plot[[y_col]], na.rm = TRUE)
      y_min <- min(df_plot[[y_col]], na.rm = TRUE)
      step <- max((y_max - y_min) * 0.10, 1)
      for (i in seq_len(nrow(sig_results))) {
        row <- sig_results[i, ]
        x1 <- match(row$Group1, dentition_order)
        x2 <- match(row$Group2, dentition_order)
        if (is.na(x1) || is.na(x2)) next
        y <- y_max + i * step
        stars <- p_to_star(row$`p-value (adjusted)`)
        p <- p +
          ggplot2::geom_segment(x = x1, xend = x2, y = y, yend = y) +
          ggplot2::geom_segment(x = x1, xend = x1, y = y, yend = y - step * 0.15) +
          ggplot2::geom_segment(x = x2, xend = x2, y = y, yend = y - step * 0.15) +
          ggplot2::annotate("text", x = (x1 + x2) / 2, y = y + step * 0.05, label = stars, size = 5)
      }
    }
  }

  if (!is.null(save_path)) ggplot2::ggsave(save_path, p, width = 10, height = 6, dpi = 300)
  invisible(p)
}

plot_abuse_by_dentition_facet_refined <- function(
  df,
  posthoc_df,
  y_col = "DMFT_Index",
  xlabel = NULL,
  ylabel = NULL,
  title = NULL,
  title_fontsize = 16,
  label_fontsize = 14,
  tick_fontsize = 12,
  save_path = NULL
) {
  dentition_order <- c("primary_dentition", "mixed_dentition", "permanent_dentition")
  preferred_abuse <- c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")
  df_plot <- df[df$dentition_type %in% dentition_order & df$abuse %in% preferred_abuse, , drop = FALSE]
  if (nrow(df_plot) == 0 || !(y_col %in% names(df_plot))) return(invisible(NULL))
  df_plot$dentition_type <- factor(df_plot$dentition_type, levels = dentition_order)
  df_plot$abuse_short <- factor(gsub(" Abuse", "", as.character(df_plot$abuse)), levels = gsub(" Abuse", "", preferred_abuse))

  p <- ggplot2::ggplot(df_plot, ggplot2::aes(x = abuse_short, y = .data[[y_col]])) +
    ggplot2::geom_boxplot(outlier.shape = NA) +
    ggplot2::geom_jitter(width = 0.15, alpha = 0.4, size = 1.3) +
    ggplot2::stat_summary(fun = mean, geom = "point", shape = 18, size = 2.5) +
    ggplot2::stat_summary(fun = mean, geom = "text", ggplot2::aes(label = sprintf("%.2f", after_stat(y))), vjust = -0.5, size = 3, fontface = "bold") +
    ggplot2::facet_wrap(~ dentition_type, nrow = 1, scales = "free_x") +
    ggplot2::labs(x = xlabel, y = ifelse(is.null(ylabel), y_col, ylabel), title = title) +
    ggplot2::theme_bw(base_size = tick_fontsize) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(face = "bold"), axis.title = ggplot2::element_text(face = "bold"), strip.text = ggplot2::element_text(face = "bold"))

  if (!is.null(save_path)) ggplot2::ggsave(save_path, p, width = 22, height = 8, dpi = 300)
  invisible(p)
}

parse_ci <- function(ci_str) {
  ci_str <- as.character(ci_str)
  m <- regexec("\\(([0-9.]+)-([0-9.]+)\\)", ci_str)
  parts <- regmatches(ci_str, m)[[1]]
  if (length(parts) == 3) return(c(as.numeric(parts[2]), as.numeric(parts[3])))
  c(NA_real_, NA_real_)
}

pairwise_mannwhitney <- function(df, var_name, group_col = "abuse", p_adjust = "bonferroni") {
  groups <- get_levels(df[[group_col]])
  pairs <- utils::combn(groups, 2, simplify = FALSE)
  n_comparisons <- length(pairs)
  results <- list()

  for (pair in pairs) {
    group1 <- pair[1]
    group2 <- pair[2]
    data1 <- safe_numeric(df[df[[group_col]] == group1, var_name])
    data2 <- safe_numeric(df[df[[group_col]] == group2, var_name])
    data1 <- data1[!is.na(data1)]
    data2 <- data2[!is.na(data2)]
    if (length(data1) == 0 || length(data2) == 0) next

    res <- stats::wilcox.test(data1, data2, alternative = "two.sided", exact = FALSE)
    u_stat <- as.numeric(res$statistic)
    p_val <- res$p.value
    r <- 1 - (2 * u_stat) / (length(data1) * length(data2))
    p_adjusted <- ifelse(p_adjust == "bonferroni", min(p_val * n_comparisons, 1), p_val)
    sig <- ifelse(p_adjusted <= 0.001, "***", ifelse(p_adjusted <= 0.01, "**", ifelse(p_adjusted <= 0.05, "*", "")))

    results[[length(results) + 1]] <- list(
      Group1 = group1,
      Group2 = group2,
      N1 = length(data1),
      N2 = length(data2),
      Median1 = sprintf("%.2f", stats::median(data1)),
      Median2 = sprintf("%.2f", stats::median(data2)),
      U_Statistic = sprintf("%.0f", u_stat),
      `p-value_raw` = format_p(p_val),
      `p-value_adjusted` = format_p(p_adjusted),
      Effect_Size_r = sprintf("%.3f", r),
      Significance = sig
    )
  }

  bind_rows_safe(results)
}

analyze_dmft_by_dentition_with_pairwise <- function(df) {
  required_cols <- c("DMFT_Index", "Present_Teeth", "Present_Baby_Teeth", "Present_Perm_Teeth", "abuse")
  for (col in required_cols) {
    if (!(col %in% names(df))) message("   Shape '", col, "' column not found in data")
  }

  df_analysis <- df
  if (!("dentition_type" %in% names(df_analysis))) {
    df_analysis$dentition_type <- apply(df_analysis, 1, function(row) {
      present_teeth <- ifelse(!is.na(row["Present_Teeth"]), as.numeric(row["Present_Teeth"]), 0)
      present_baby <- ifelse(!is.na(row["Present_Baby_Teeth"]), as.numeric(row["Present_Baby_Teeth"]), 0)
      present_perm <- ifelse(!is.na(row["Present_Perm_Teeth"]), as.numeric(row["Present_Perm_Teeth"]), 0)
      if (present_teeth == 0) return("No_Teeth")
      if (present_baby == present_teeth && present_perm == 0) return("primary_dentition")
      if (present_perm == present_teeth && present_baby == 0) return("permanent_dentition")
      "mixed_dentition"
    })
  }

  dentition_order <- c("primary_dentition", "mixed_dentition", "permanent_dentition")
  all_pairwise_results <- list()

  for (dent_type in dentition_order) {
    df_subset <- df_analysis[df_analysis$dentition_type == dent_type, , drop = FALSE]
    if (nrow(df_subset) < 10) next
    pairwise_df <- pairwise_mannwhitney(df_subset, "DMFT_Index", "abuse", "bonferroni")
    if (nrow(pairwise_df) > 0) {
      pairwise_df <- cbind(Dentition_Type = dent_type, pairwise_df)
      all_pairwise_results[[length(all_pairwise_results) + 1]] <- pairwise_df
    }
  }

  if (length(all_pairwise_results) > 0) dplyr::bind_rows(all_pairwise_results) else data.frame()
}

# -----------------------------------------------------------------------------
# Table 7: DMFT, Dt, Mt, Ft by year and abuse type
# -----------------------------------------------------------------------------

create_table_dmft_by_year_abuse <- function(df) {
  df_local <- df
  if ("date" %in% names(df_local) && !inherits(df_local$date, c("Date", "POSIXct", "POSIXt"))) {
    df_local$date <- as.Date(df_local$date)
  }
  if (!("year" %in% names(df_local)) && "date" %in% names(df_local)) {
    df_local$year <- as.integer(format(df_local$date, "%Y"))
  } else if (!("year" %in% names(df_local))) {
    message("   ⚠ 'year' or 'date' column not found")
    return(data.frame())
  }

  df_local$Dt <- add_fill0(df_local$Perm_D, df_local$Baby_d)
  df_local$Mt <- add_fill0(df_local$Perm_M, df_local$Baby_m)
  df_local$Ft <- add_fill0(df_local$Perm_F, df_local$Baby_f)
  df_local$DFt <- add_fill0(df_local$Dt, df_local$Ft)

  abuse_types <- get_levels(df_local$abuse)
  years <- sort(unique(df_local$year[!is.na(df_local$year)]))
  results <- list()

  vars_to_summarize <- list(
    c("DMFT_Index", "DMFT"),
    c("Perm_DMFT", "Perm_DMFT"),
    c("Baby_DMFT", "Baby_DMFT"),
    c("Dt", "Dt (Untreated)"),
    c("Mt", "Mt (Missing)"),
    c("Ft", "Ft (Filled)"),
    c("DFt", "DFt (Dt+Ft)")
  )

  for (year in years) {
    df_year <- df_local[df_local$year == year, , drop = FALSE]
    groups_dmft <- lapply(abuse_types, function(abuse) safe_numeric(df_year[df_year$abuse == abuse, "DMFT_Index"]))
    groups_dmft <- lapply(groups_dmft, function(g) g[!is.na(g)])
    groups_dmft <- groups_dmft[vapply(groups_dmft, length, integer(1)) > 0]
    p_kw <- safe_kruskal_p(groups_dmft)
    p_val_str <- ifelse(is.na(p_kw), "N/A", format_p(p_kw))

    first_row <- TRUE
    for (abuse in abuse_types) {
      subset <- df_year[df_year$abuse == abuse, , drop = FALSE]
      n <- nrow(subset)
      if (n == 0) next
      row <- list(Year = ifelse(first_row, as.integer(year), ""), Abuse_Type = abuse, N = n)
      for (var_pair in vars_to_summarize) {
        var_col <- var_pair[1]
        var_name <- var_pair[2]
        data <- safe_numeric(subset[[var_col]])
        data <- data[!is.na(data)]
        if (length(data) > 0) {
          row[[paste0(var_name, " Mean (SD)")]] <- sprintf("%.2f (%.2f)", mean(data), safe_sd(data))
          q <- stats::quantile(data, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
          row[[paste0(var_name, " Median [IQR]")]] <- sprintf("%.1f [%.1f-%.1f]", stats::median(data), q[1], q[2])
        } else {
          row[[paste0(var_name, " Mean (SD)")]] <- "N/A"
          row[[paste0(var_name, " Median [IQR]")]] <- "N/A"
        }
      }
      row[["DMFT p-value (KW)"]] <- ifelse(first_row, p_val_str, "")
      results[[length(results) + 1]] <- row
      first_row <- FALSE
    }
  }

  empty_row <- list(Year = "=== OVERALL BY YEAR ===", Abuse_Type = "(Combined)", N = "---")
  for (var_pair in vars_to_summarize) {
    var_name <- var_pair[2]
    empty_row[[paste0(var_name, " Mean (SD)")]] <- "---"
    empty_row[[paste0(var_name, " Median [IQR]")]] <- "---"
  }
  empty_row[["DMFT p-value (KW)"]] <- "---"
  results[[length(results) + 1]] <- empty_row

  year_groups <- lapply(years, function(y) safe_numeric(df_local[df_local$year == y, "DMFT_Index"]))
  year_groups <- lapply(year_groups, function(g) g[!is.na(g)])
  year_groups <- year_groups[vapply(year_groups, length, integer(1)) > 0]
  p_kw_year <- safe_kruskal_p(year_groups)
  p_val_year_str <- ifelse(is.na(p_kw_year), "N/A", format_p(p_kw_year))

  first_year <- TRUE
  for (year in years) {
    subset <- df_local[df_local$year == year, , drop = FALSE]
    n <- nrow(subset)
    if (n > 0) {
      subset_dmft <- safe_numeric(subset$DMFT_Index)
      subset_dmft <- subset_dmft[!is.na(subset_dmft)]
      row <- list(Year = as.integer(year), Abuse_Type = "All abuse types", N = n)
      if (length(subset_dmft) > 0) {
        q <- stats::quantile(subset_dmft, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
        row[["Mean"]] <- sprintf("%.2f", mean(subset_dmft))
        row[["SD"]] <- sprintf("%.2f", safe_sd(subset_dmft))
        row[["Median"]] <- sprintf("%.1f", stats::median(subset_dmft))
        row[["25%"]] <- sprintf("%.1f", q[1])
        row[["75%"]] <- sprintf("%.1f", q[2])
        row[["Min"]] <- sprintf("%.0f", min(subset_dmft))
        row[["Max"]] <- sprintf("%.0f", max(subset_dmft))
      } else {
        row[["Mean"]] <- "N/A"; row[["SD"]] <- "N/A"; row[["Median"]] <- "N/A"
        row[["25%"]] <- "N/A"; row[["75%"]] <- "N/A"; row[["Min"]] <- "N/A"; row[["Max"]] <- "N/A"
      }
      for (var_pair in vars_to_summarize) {
        var_col <- var_pair[1]
        var_name <- var_pair[2]
        data <- safe_numeric(subset[[var_col]])
        data <- data[!is.na(data)]
        if (length(data) > 0) {
          q <- stats::quantile(data, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
          row[[paste0(var_name, " Mean (SD)")]] <- sprintf("%.2f (%.2f)", mean(data), safe_sd(data))
          row[[paste0(var_name, " Median [IQR]")]] <- sprintf("%.1f [%.1f-%.1f]", stats::median(data), q[1], q[2])
        } else {
          row[[paste0(var_name, " Mean (SD)")]] <- "N/A"
          row[[paste0(var_name, " Median [IQR]")]] <- "N/A"
        }
      }
      row[["DMFT p-value (KW)"]] <- ifelse(first_year, p_val_year_str, "")
      results[[length(results) + 1]] <- row
      first_year <- FALSE
    }
  }

  bind_rows_safe(results)
}

# -----------------------------------------------------------------------------
# Forest plot and general visualizations
# -----------------------------------------------------------------------------

create_forest_plot_vertical <- function(df_logistic, df_original, output_dir, timestamp, figsize = c(10, 10)) {
  df <- df_logistic
  if (nrow(df) == 0) return(invisible(NULL))
  if ("Stratum" %in% names(df)) df <- df[df$Stratum %in% c("", "Overall"), , drop = FALSE]
  df <- df[df$`Odds Ratio` != "N/A", , drop = FALSE]
  if (nrow(df) == 0) return(invisible(NULL))

  df$OR <- safe_numeric(df$`Odds Ratio`)
  cis <- t(vapply(df$`95% CI`, parse_ci, numeric(2)))
  df$CI_lower <- cis[, 1]
  df$CI_upper <- cis[, 2]
  df$p_numeric <- vapply(df$`p-value`, p_value_to_numeric, numeric(1))
  df$significant <- df$p_numeric < 0.05
  df$label <- paste0(gsub(" vs Physical Abuse", "", df$Comparison), "\n", df$Outcome)

  p <- ggplot2::ggplot(df, ggplot2::aes(x = OR, y = reorder(label, OR))) +
    ggplot2::geom_vline(xintercept = 1, linetype = "dashed") +
    ggplot2::geom_errorbarh(ggplot2::aes(xmin = CI_lower, xmax = CI_upper), height = 0.2) +
    ggplot2::geom_point(ggplot2::aes(shape = significant), size = 3) +
    ggplot2::geom_text(ggplot2::aes(label = sprintf("%.2f (%.2f-%.2f)", OR, CI_lower, CI_upper)), hjust = -0.05, size = 3) +
    ggplot2::labs(x = "Odds Ratio (95% CI)", y = NULL) +
    ggplot2::theme_bw()

  ggplot2::ggsave(file.path(output_dir, paste0("figure_forest_plot_", timestamp, ".png")), p, width = figsize[1], height = figsize[2], dpi = 300)
  invisible(p)
}

create_visualizations <- function(df, output_dir) {
  df_plot <- df
  abuse_order <- c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")
  df_plot <- df_plot[df_plot$abuse %in% abuse_order, , drop = FALSE]
  if (nrow(df_plot) == 0) return(invisible(NULL))
  df_plot$abuse <- factor(as.character(df_plot$abuse), levels = abuse_order)

  if ("DMFT_Index" %in% names(df_plot)) {
    p <- ggplot2::ggplot(df_plot, ggplot2::aes(x = abuse, y = DMFT_Index)) +
      ggplot2::geom_boxplot(outlier.shape = NA) +
      ggplot2::labs(x = NULL, y = "DMFT Index") +
      ggplot2::theme_bw()
    ggplot2::ggsave(file.path(output_dir, "figure1_dmft_boxplot.png"), p, width = 10, height = 6, dpi = 300)
  }

  cat_vars <- list(
    c("gingivitis", "Gingivitis"),
    c("needTOBEtreated", "Treatment Need"),
    c("OralCleanStatus", "Oral Hygiene Status")
  )

  for (var_info in cat_vars) {
    var_name <- var_info[1]
    var_label <- var_info[2]
    if (!(var_name %in% names(df_plot))) next
    df_valid <- df_plot[!is.na(df_plot[[var_name]]), , drop = FALSE]
    if (nrow(df_valid) == 0) next
    pct_df <- df_valid %>%
      dplyr::count(abuse, .data[[var_name]], name = "n") %>%
      dplyr::group_by(abuse) %>%
      dplyr::mutate(pct = n / sum(n) * 100) %>%
      dplyr::ungroup()
    p <- ggplot2::ggplot(pct_df, ggplot2::aes(x = abuse, y = pct, fill = .data[[var_name]])) +
      ggplot2::geom_col(position = "stack") +
      ggplot2::labs(x = NULL, y = "Percentage (%)", fill = var_label) +
      ggplot2::theme_bw()
    ggplot2::ggsave(file.path(output_dir, paste0("figure_", var_name, "_bar.png")), p, width = 10, height = 6, dpi = 300)
  }

  invisible(NULL)
}

plot_boxplot_with_dunn <- function(
  df,
  var_name,
  group_col = "abuse",
  xlabel = NULL,
  ylabel = NULL,
  title = NULL,
  title_fontsize = 14,
  label_fontsize = 14,
  tick_fontsize = 12,
  output_dir = NULL,
  p_adjust = "bonferroni"
) {
  if (is.null(output_dir)) output_dir <- "./"
  timestamp_local <- format(Sys.Date(), "%Y%m%d")
  ratio_vars <- c("Care_Index", "UTN_Score")
  cols <- c(group_col, var_name)
  if (var_name %in% ratio_vars && "DMFT_Index" %in% names(df)) cols <- c(cols, "DMFT_Index")
  data <- df[, unique(cols), drop = FALSE]
  data <- data[!is.na(data[[group_col]]) & !is.na(data[[var_name]]), , drop = FALSE]
  if (var_name %in% ratio_vars && "DMFT_Index" %in% names(data)) data <- data[data$DMFT_Index > 0, , drop = FALSE]
  if (nrow(data) == 0) return(invisible(NULL))

  if (group_col == "abuse") {
    preferred_order <- c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")
    categories <- c(preferred_order[preferred_order %in% unique(as.character(data[[group_col]]))], setdiff(sort(unique(as.character(data[[group_col]]))), preferred_order))
  } else {
    categories <- sort(unique(as.character(data[[group_col]])))
  }

  data[[group_col]] <- factor(as.character(data[[group_col]]), levels = categories)
  dunn <- tryCatch(dunn_posthoc(data, var_name, group_col, p_adjust), error = function(e) NULL)
  if (is.null(dunn)) return(invisible(NULL))

  p <- ggplot2::ggplot(data, ggplot2::aes(x = .data[[group_col]], y = .data[[var_name]])) +
    ggplot2::geom_boxplot(outlier.shape = NA) +
    ggplot2::geom_jitter(width = 0.15, alpha = 0.4, size = 1.5) +
    ggplot2::stat_summary(fun = mean, geom = "point", shape = 18, size = 3) +
    ggplot2::stat_summary(fun = mean, geom = "text", ggplot2::aes(label = sprintf("%.2f", after_stat(y))), vjust = -0.5, fontface = "bold") +
    ggplot2::labs(x = xlabel, y = ifelse(is.null(ylabel), var_name, ylabel), title = title) +
    ggplot2::theme_bw(base_size = tick_fontsize) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(face = "bold"), axis.title = ggplot2::element_text(face = "bold"))

  y_max <- max(data[[var_name]], na.rm = TRUE)
  y_min <- min(data[[var_name]], na.rm = TRUE)
  h_step <- max((y_max - y_min) * 0.10, 1)
  y_start <- y_max + h_step * 0.5
  sig_count <- 0
  for (pair in utils::combn(categories, 2, simplify = FALSE)) {
    p_val <- dunn$adjusted[pair[1], pair[2]]
    if (!is.na(p_val) && p_val < 0.05) {
      x1 <- match(pair[1], categories)
      x2 <- match(pair[2], categories)
      y <- y_start + sig_count * h_step
      stars <- p_to_star(p_val)
      p <- p +
        ggplot2::geom_segment(x = x1, xend = x2, y = y, yend = y) +
        ggplot2::geom_segment(x = x1, xend = x1, y = y, yend = y - h_step * 0.2) +
        ggplot2::geom_segment(x = x2, xend = x2, y = y, yend = y - h_step * 0.2) +
        ggplot2::annotate("text", x = (x1 + x2) / 2, y = y + h_step * 0.05, label = stars, size = 5)
      sig_count <- sig_count + 1
    }
  }

  ggplot2::ggsave(file.path(output_dir, paste0("pairwise_results_", var_name, "_", timestamp_local, ".png")), p, width = 10, height = 6, dpi = 300)
  invisible(p)
}

plot_boxplot_by_dentition_type <- function(
  df,
  xlabel = NULL,
  ylabel = NULL,
  title = NULL,
  title_fontsize = 14,
  label_fontsize = 14,
  tick_fontsize = 12,
  output_dir = NULL,
  p_adjust = "bonferroni"
) {
  if (is.null(output_dir)) output_dir <- "./"
  timestamp_local <- format(Sys.Date(), "%Y%m%d")
  df_analysis <- df

  if (!("dentition_type" %in% names(df_analysis))) {
    df_analysis$dentition_type <- apply(df_analysis, 1, function(row) {
      present_teeth <- ifelse(!is.na(row["total_teeth"]), as.numeric(row["total_teeth"]), 0)
      present_baby <- ifelse(!is.na(row["Baby_total_teeth"]), as.numeric(row["Baby_total_teeth"]), 0)
      present_perm <- ifelse(!is.na(row["Perm_total_teeth"]), as.numeric(row["Perm_total_teeth"]), 0)
      if (present_teeth == 0) return("No_Teeth")
      if (present_baby == present_teeth && present_perm == 0) return("primary_dentition")
      if (present_perm == present_teeth && present_baby == 0) return("permanent_dentition")
      "mixed_dentition"
    })
  }

  dentition_order <- c("primary_dentition", "mixed_dentition", "permanent_dentition")
  data <- df_analysis[df_analysis$dentition_type %in% dentition_order & !is.na(df_analysis$DMFT_Index), , drop = FALSE]
  if (nrow(data) == 0) return(invisible(NULL))
  data$dentition_type <- factor(data$dentition_type, levels = dentition_order)
  dunn <- tryCatch(dunn_posthoc(data, "DMFT_Index", "dentition_type", p_adjust), error = function(e) NULL)
  if (is.null(dunn)) return(invisible(NULL))

  p <- ggplot2::ggplot(data, ggplot2::aes(x = dentition_type, y = DMFT_Index)) +
    ggplot2::geom_boxplot(outlier.shape = NA) +
    ggplot2::geom_jitter(width = 0.15, alpha = 0.4, size = 1.5) +
    ggplot2::stat_summary(fun = mean, geom = "point", shape = 18, size = 3) +
    ggplot2::stat_summary(fun = mean, geom = "text", ggplot2::aes(label = sprintf("%.2f", after_stat(y))), vjust = -0.5, fontface = "bold") +
    ggplot2::labs(x = xlabel, y = ifelse(is.null(ylabel), "DMFT Index", ylabel), title = title) +
    ggplot2::theme_bw(base_size = tick_fontsize)

  y_max <- max(data$DMFT_Index, na.rm = TRUE)
  y_min <- min(data$DMFT_Index, na.rm = TRUE)
  h_step <- max((y_max - y_min) * 0.10, 1)
  y_start <- y_max + h_step * 0.5
  sig_count <- 0
  for (pair in utils::combn(dentition_order, 2, simplify = FALSE)) {
    p_val <- dunn$adjusted[pair[1], pair[2]]
    if (!is.na(p_val) && p_val < 0.05) {
      x1 <- match(pair[1], dentition_order)
      x2 <- match(pair[2], dentition_order)
      y <- y_start + sig_count * h_step
      stars <- p_to_star(p_val)
      p <- p +
        ggplot2::geom_segment(x = x1, xend = x2, y = y, yend = y) +
        ggplot2::geom_segment(x = x1, xend = x1, y = y, yend = y - h_step * 0.2) +
        ggplot2::geom_segment(x = x2, xend = x2, y = y, yend = y - h_step * 0.2) +
        ggplot2::annotate("text", x = (x1 + x2) / 2, y = y + h_step * 0.05, label = stars, size = 5)
      sig_count <- sig_count + 1
    }
  }

  ggplot2::ggsave(file.path(output_dir, paste0("pairwise_results_dentition_type_", timestamp_local, ".png")), p, width = 10, height = 6, dpi = 300)
  invisible(p)
}

generate_summary_report <- function(df, table3_overall, output_dir, timestamp) {
  output_path <- file.path(output_dir, paste0("summary_report_", timestamp, ".txt"))
  con <- file(output_path, open = "w")
  on.exit(close(con), add = TRUE)
  writeLines("Summary Report", con)
  writeLines(paste0("Total N: ", nrow(df)), con)
  if (!is.null(table3_overall) && nrow(table3_overall) > 0) {
    sig <- table3_overall[table3_overall$Significant == "Yes", , drop = FALSE]
    writeLines("Significant Differences:", con)
    utils::capture.output(print(sig), file = con, append = TRUE)
  }
  message("Summary saved to ", output_dir)
  invisible(NULL)
}
