# ============================================================================
# Refactored Analysis Code (REVISED) - R translation
# Translated from Analysis_code_refactored_REVISED(6).py
# - Enforces a single index observation per child (if an ID column exists)
# - Adds robust handling for ratio indices (Care_Index / UTN_Score)
# - Adds year variable for optional year fixed effects in regression
# - Adds sensitivity analysis including multi-type maltreatment records (abuse_num > 1)
# - Adds stratified logistic regression by dentition_type
# ============================================================================

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- args[grepl("^--file=", args)]
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg[1]))))
  }
  frame_files <- vapply(sys.frames(), function(x) {
    if (!is.null(x$ofile)) return(as.character(x$ofile))
    NA_character_
  }, character(1))
  frame_files <- frame_files[!is.na(frame_files)]
  if (length(frame_files) > 0) {
    return(dirname(normalizePath(frame_files[length(frame_files)])))
  }
  getwd()
}

SCRIPT_DIR <- get_script_dir()
FUNCTIONS_PATH <- file.path(SCRIPT_DIR, "Functions_refactored_REVISED_R.R")
if (!file.exists(FUNCTIONS_PATH)) {
  FUNCTIONS_PATH <- file.path(getwd(), "Functions_refactored_REVISED_R.R")
}
source(FUNCTIONS_PATH)

log_info <- function(...) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - ", paste(..., collapse = ""))
}

log_error <- function(...) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - ERROR - ", paste(..., collapse = ""))
}

_pick_first_existing_col <- function(df, candidates) {
  for (c in candidates) {
    if (c %in% names(df)) return(c)
  }
  NULL
}

_deduplicate_to_first_exam <- function(df, id_col) {
  if (is.null(id_col) || !(id_col %in% names(df)) || !("date" %in% names(df))) return(df)

  df_sorted <- df[order(df$date), , drop = FALSE]
  before <- nrow(df_sorted)
  df_dedup <- df_sorted[!duplicated(df_sorted[[id_col]]), , drop = FALSE]
  after <- nrow(df_dedup)
  log_info("Deduplication by ", id_col, ": ", before, " -> ", after, " rows (kept first exam date).")

  df_dedup
}

summary_profile <- function(df) {
  rows <- lapply(names(df), function(col) {
    x <- df[[col]]
    if (is.numeric(x)) {
      data.frame(
        Variable = col,
        N = sum(!is.na(x)),
        Missing = sum(is.na(x)),
        Mean = mean(x, na.rm = TRUE),
        SD = stats::sd(x, na.rm = TRUE),
        Min = min(x, na.rm = TRUE),
        Q1 = stats::quantile(x, 0.25, na.rm = TRUE, names = FALSE),
        Median = stats::median(x, na.rm = TRUE),
        Q3 = stats::quantile(x, 0.75, na.rm = TRUE, names = FALSE),
        Max = max(x, na.rm = TRUE),
        stringsAsFactors = FALSE
      )
    } else {
      data.frame(
        Variable = col,
        N = sum(!is.na(x)),
        Missing = sum(is.na(x)),
        Mean = NA_real_, SD = NA_real_, Min = NA_real_, Q1 = NA_real_,
        Median = NA_real_, Q3 = NA_real_, Max = NA_real_,
        stringsAsFactors = FALSE
      )
    }
  })
  dplyr::bind_rows(rows)
}

_engineer_oral_health_variables <- function(df) {
  df <- df

  # Age group
  if ("age_year" %in% names(df)) {
    df$age_group <- cut(
      df$age_year,
      breaks = c(0, 6, 12, 18),
      labels = c("Early Childhood (2-6)", "Middle Childhood (7-12)", "Adolescence (13-18)"),
      right = TRUE,
      include.lowest = TRUE
    )
  }

  # Teeth columns, matching the Python naming pattern.
  perm_teeth_cols <- c(
    unlist(lapply(c(1, 2), function(i) paste0("U", i, 1:7))),
    unlist(lapply(c(3, 4), function(i) paste0("L", i, 1:7)))
  )
  baby_teeth_cols <- c(
    unlist(lapply(c(5, 6), function(i) paste0("u", i, 1:5))),
    unlist(lapply(c(7, 8), function(i) paste0("l", i, 1:5)))
  )

  perm_cols <- perm_teeth_cols[perm_teeth_cols %in% names(df)]
  baby_cols <- baby_teeth_cols[baby_teeth_cols %in% names(df)]

  # Coding:
  # -1: unerupted, 0: sound, 1: filled, 2: C0, 3: caries, 4: missing,
  # 5: other/supernumerary, 6: congenital missing, 7: trauma, 8: retained deciduous, 9: fused tooth.
  if (length(perm_cols) > 0) {
    all_nan_mask_perm <- apply(is.na(df[, perm_cols, drop = FALSE]), 1, all)

    df$Perm_D <- rowSums(df[, perm_cols, drop = FALSE] == 3, na.rm = TRUE)
    df$Perm_D[all_nan_mask_perm] <- NA_real_
    df$Perm_M <- rowSums(df[, perm_cols, drop = FALSE] == 4, na.rm = TRUE)
    df$Perm_M[all_nan_mask_perm] <- NA_real_
    df$Perm_F <- rowSums(df[, perm_cols, drop = FALSE] == 1, na.rm = TRUE)
    df$Perm_F[all_nan_mask_perm] <- NA_real_
    df$Perm_Sound <- rowSums(df[, perm_cols, drop = FALSE] == 0, na.rm = TRUE)
    df$Perm_Sound[all_nan_mask_perm] <- NA_real_
    df$Perm_DMFT <- df$Perm_D + df$Perm_M + df$Perm_F
    df$Perm_C0 <- rowSums(df[, perm_cols, drop = FALSE] == 2, na.rm = TRUE)
    df$Perm_C0[all_nan_mask_perm] <- NA_real_
    df$Perm_DMFT_C0 <- df$Perm_DMFT + df$Perm_C0
    df$Perm_total_teeth <- rowSums(!is.na(df[, perm_cols, drop = FALSE]) & df[, perm_cols, drop = FALSE] != -1, na.rm = TRUE)
    df$Perm_sound_rate <- (df$Perm_Sound / df$Perm_total_teeth * 100)
    df$Perm_sound_rate[is.infinite(df$Perm_sound_rate)] <- NA_real_
  } else {
    for (col in c("Perm_D", "Perm_M", "Perm_F", "Perm_Sound", "Perm_DMFT", "Perm_C0", "Perm_DMFT_C0", "Perm_sound_rate")) {
      df[[col]] <- NA_real_
    }
    df$Perm_total_teeth <- 0
  }

  if (length(baby_cols) > 0) {
    all_nan_mask_baby <- apply(is.na(df[, baby_cols, drop = FALSE]), 1, all)

    df$Baby_d <- rowSums(df[, baby_cols, drop = FALSE] == 3, na.rm = TRUE)
    df$Baby_d[all_nan_mask_baby] <- NA_real_
    df$Baby_m <- rowSums(df[, baby_cols, drop = FALSE] == 4, na.rm = TRUE)
    df$Baby_m[all_nan_mask_baby] <- NA_real_
    df$Baby_f <- rowSums(df[, baby_cols, drop = FALSE] == 1, na.rm = TRUE)
    df$Baby_f[all_nan_mask_baby] <- NA_real_
    df$Baby_sound <- rowSums(df[, baby_cols, drop = FALSE] == 0, na.rm = TRUE)
    df$Baby_sound[all_nan_mask_baby] <- NA_real_
    df$Baby_DMFT <- df$Baby_d + df$Baby_m + df$Baby_f
    df$Baby_C0 <- rowSums(df[, baby_cols, drop = FALSE] == 2, na.rm = TRUE)
    df$Baby_C0[all_nan_mask_baby] <- NA_real_
    df$Baby_DMFT_C0 <- df$Baby_DMFT + df$Baby_C0
    df$Baby_total_teeth <- rowSums(!is.na(df[, baby_cols, drop = FALSE]) & df[, baby_cols, drop = FALSE] != -1, na.rm = TRUE)
    df$Baby_sound_rate <- (df$Baby_sound / df$Baby_total_teeth * 100)
    df$Baby_sound_rate[is.infinite(df$Baby_sound_rate)] <- NA_real_
  } else {
    for (col in c("Baby_d", "Baby_m", "Baby_f", "Baby_sound", "Baby_DMFT", "Baby_C0", "Baby_DMFT_C0", "Baby_sound_rate")) {
      df[[col]] <- NA_real_
    }
    df$Baby_total_teeth <- 0
  }

  # Total DMFT/dmft values.
  df$DMFT_Index <- add_fill0(df$Perm_DMFT, df$Baby_DMFT)
  df$DMFT_C0 <- add_fill0(df$Perm_DMFT_C0, df$Baby_DMFT_C0)
  df$C0_Count <- add_fill0(df$Perm_C0, df$Baby_C0)

  # Indices are undefined when DMFT_Index == 0.
  denom <- as.numeric(df$DMFT_Index)
  df$filled_total <- add_fill0(df$Perm_F, df$Baby_f)
  df$decayed_total <- add_fill0(df$Perm_D, df$Baby_d)
  df$missing_total <- add_fill0(df$Perm_M, df$Baby_m)

  df$Care_Index <- df$filled_total / denom * 100
  df$Care_Index[is.infinite(df$Care_Index) | denom <= 0] <- NA_real_

  df$UTN_Score <- df$decayed_total / denom * 100
  df$UTN_Score[is.infinite(df$UTN_Score) | denom <= 0] <- NA_real_

  df$total_teeth <- add_fill0(df$Perm_total_teeth, df$Baby_total_teeth)
  df$Healthy_Rate <- add_fill0(df$Perm_Sound, df$Baby_sound) / df$total_teeth * 100
  df$Healthy_Rate[is.infinite(df$Healthy_Rate) | df$total_teeth <= 0] <- NA_real_

  # Aliases for downstream functions.
  df$Present_Teeth <- df$total_teeth
  df$Present_Perm_Teeth <- df$Perm_total_teeth
  df$Present_Baby_Teeth <- df$Baby_total_teeth

  # Binary outcomes.
  df$has_caries <- as.integer(!is.na(df$DMFT_Index) & df$DMFT_Index > 0)
  df$has_untreated_caries <- as.integer(!is.na(df$decayed_total) & df$decayed_total > 0)

  # Dentition type. Retained deciduous teeth are treated as mixed dentition.
  df$dentition_type <- mapply(function(present_teeth, present_baby, present_perm) {
    present_teeth <- ifelse(is.na(present_teeth), 0, present_teeth)
    present_baby <- ifelse(is.na(present_baby), 0, present_baby)
    present_perm <- ifelse(is.na(present_perm), 0, present_perm)
    if (present_teeth == 0) return("No_Teeth")
    if (present_baby == present_teeth && present_perm == 0) return("primary_dentition")
    if (present_perm == present_teeth && present_baby == 0) return("permanent_dentition")
    "mixed_dentition"
  }, df$total_teeth, df$Baby_total_teeth, df$Perm_total_teeth)

  if ("date" %in% names(df) && inherits(df$date, c("Date", "POSIXct", "POSIXt"))) {
    df$year <- as.integer(format(df$date, "%Y"))
  }

  # The Python version wrote a debug CSV to an absolute user path. This R translation
  # intentionally avoids that hard-coded path and writes only to configured output folders.
  df
}

main <- function() {
  log_info("Starting Analysis...")
  timestamp <- format(Sys.Date(), "%Y%m%d")

  # ============================================================================
  # Configuration
  # ============================================================================
  BASE_DIR <- dirname(SCRIPT_DIR)  # assumes code/ is one level deep; override below if needed
  DATA_DIR <- file.path(BASE_DIR, "data")
  DATA_DESCRIPTION_OUTPUT_DIR <- file.path(DATA_DIR, "data_description")
  OUTPUT_DIR <- paste0(file.path(BASE_DIR, "result", timestamp), .Platform$file.sep)

  dir.create(DATA_DESCRIPTION_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
  print(paste0("OUTPUT_DIR: ", OUTPUT_DIR))

  ORIGINAL_DATA_NAME <- "analysisData_20260211"
  ORIGINAL_DATA_PATH <- file.path(DATA_DIR, paste0(ORIGINAL_DATA_NAME, ".csv"))

  END_DATE <- as.Date("2024-03-31")
  target_abuse_types <- c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")

  SUBJECT_ID_COL_CANDIDATES <- c("No_All", "child_id", "subject_id", "case_id", "ID", "id")
  EXAMINER_COL_CANDIDATES <- c("dentist", "examiner", "doctor", "operator", "checker")

  # ============================================================================
  # Data Loading
  # ============================================================================
  log_info("Loading data from ", ORIGINAL_DATA_PATH)
  if (!file.exists(ORIGINAL_DATA_PATH)) {
    log_error("Data file not found: ", ORIGINAL_DATA_PATH)
    return(invisible(NULL))
  }

  data0 <- utils::read.csv(ORIGINAL_DATA_PATH, stringsAsFactors = FALSE, check.names = FALSE)
  log_info("Loaded data shape: ", nrow(data0), " x ", ncol(data0))

  # Save columns.
  writeLines(names(data0), file.path(DATA_DESCRIPTION_OUTPUT_DIR, paste0(ORIGINAL_DATA_NAME, "_colnames.txt")))

  # ============================================================================
  # Preprocessing
  # ============================================================================
  if ("date" %in% names(data0)) {
    data0$date <- as.Date(data0$date)
  }

  mappings <- list(
    abuse = c(
      "1" = "Physical Abuse", "2" = "Neglect", "3" = "Emotional Abuse", "4" = "Sexual Abuse",
      "5" = "Delinquency", "6" = "Parenting Difficulties", "7" = "Others"
    ),
    occlusalRelationship = c(
      "1" = "Normal Occlusion", "2" = "Crowding", "3" = "Anterior Crossbite", "4" = "Open Bite",
      "5" = "Maxillary Protrusion", "6" = "Crossbite", "7" = "Others"
    ),
    needTOBEtreated = c("1" = "No Treatment Required", "2" = "Treatment Required"),
    emergency = c("1" = "Urgent Treatment Required"),
    gingivitis = c("1" = "No Gingivitis", "2" = "Gingivitis"),
    OralCleanStatus = c("1" = "Poor", "2" = "Fair", "3" = "Good"),
    habits = c("1" = "None", "2" = "Digit Sucking", "3" = "Nail biting", "4" = "Tongue Thrusting", "5" = "Smoking", "6" = "Others")
  )

  for (col in names(mappings)) {
    if (col %in% names(data0)) {
      x <- as.character(data0[[col]])
      mapped <- mappings[[col]][x]
      data0[[col]] <- ifelse(!is.na(mapped), unname(mapped), x)
    }
  }

  orders <- list(
    abuse = c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse", "Delinquency", "Parenting Difficulties", "Others"),
    occlusalRelationship = c("Normal Occlusion", "Crowding", "Anterior Crossbite", "Open Bite", "Maxillary Protrusion", "Crossbite", "Others"),
    needTOBEtreated = c("No Treatment Required", "Treatment Required"),
    emergency = c("Urgent Treatment Required"),
    gingivitis = c("No Gingivitis", "Gingivitis"),
    OralCleanStatus = c("Poor", "Fair", "Good"),
    habits = c("None", "Digit Sucking", "Nail biting", "Tongue Thrusting", "Smoking", "Others")
  )

  for (col in names(orders)) {
    if (col %in% names(data0)) {
      data0[[col]] <- factor(data0[[col]], levels = orders[[col]], ordered = TRUE)
    }
  }

  cleaned_path <- file.path(DATA_DIR, paste0(ORIGINAL_DATA_NAME, "_AllData_cleaned.csv"))
  utils::write.csv(data0, cleaned_path, row.names = FALSE)

  save_value_counts_summary(
    data0,
    file.path(DATA_DESCRIPTION_OUTPUT_DIR, paste0("unique_values_summary_", ORIGINAL_DATA_NAME, ".csv")),
    exclude_cols = c("No_All", "instruction_detail", "instruction", "memo")
  )
  utils::write.csv(summary_profile(data0), file.path(DATA_DESCRIPTION_OUTPUT_DIR, paste0(ORIGINAL_DATA_NAME, "_description.csv")), row.names = FALSE)

  # ============================================================================
  # Filtering & flow accounting
  # ============================================================================
  log_info("Filtering data...")
  df_date <- data0
  if ("date" %in% names(df_date)) {
    df_date <- df_date[df_date$date <= END_DATE & !is.na(df_date$date), , drop = FALSE]
  }

  if ("abuse_num" %in% names(df_date)) {
    df_all <- df_date[df_date$abuse_num >= 1 & df_date$abuse %in% target_abuse_types, , drop = FALSE]
  } else {
    df_all <- df_date[df_date$abuse %in% target_abuse_types, , drop = FALSE]
  }

  if ("abuse_num" %in% names(df_all)) {
    df_main <- df_all[df_all$abuse_num == 1, , drop = FALSE]
  } else {
    df_main <- df_all
  }

  subject_id_col <- _pick_first_existing_col(df_main, SUBJECT_ID_COL_CANDIDATES)
  examiner_col <- _pick_first_existing_col(df_main, EXAMINER_COL_CANDIDATES)

  df_main <- _deduplicate_to_first_exam(df_main, subject_id_col)

  if ("abuse" %in% names(df_main) && is.factor(df_main$abuse)) {
    df_main$abuse <- droplevels(df_main$abuse)
  }

  log_info("Main dataset shape (single-type + dedup if possible): ", nrow(df_main), " x ", ncol(df_main))

  csv_name <- paste0(ORIGINAL_DATA_NAME, "_tillMar2024_singleType_dedup")
  utils::write.csv(df_main, file.path(DATA_DIR, paste0(csv_name, ".csv")), row.names = FALSE)

  flow_rows <- list(
    data.frame(Step = "Loaded raw", N = nrow(data0), stringsAsFactors = FALSE),
    data.frame(Step = paste0("Date <= ", END_DATE), N = nrow(df_date), stringsAsFactors = FALSE),
    data.frame(Step = "Target maltreatment (abuse in 4 types) & abuse_num>=1", N = nrow(df_all), stringsAsFactors = FALSE)
  )

  if ("abuse_num" %in% names(df_all)) {
    flow_rows[[length(flow_rows) + 1]] <- data.frame(Step = "Single-type only (abuse_num==1)", N = sum(df_all$abuse_num == 1, na.rm = TRUE), stringsAsFactors = FALSE)
    flow_rows[[length(flow_rows) + 1]] <- data.frame(Step = "Multi-type excluded (abuse_num>1)", N = sum(df_all$abuse_num > 1, na.rm = TRUE), stringsAsFactors = FALSE)
  }

  if (!is.null(subject_id_col)) {
    flow_rows[[length(flow_rows) + 1]] <- data.frame(Step = paste0("Deduplicated to first exam per ", subject_id_col), N = nrow(df_main), stringsAsFactors = FALSE)
  }

  utils::write.csv(dplyr::bind_rows(flow_rows), file.path(OUTPUT_DIR, paste0("flow_summary_", timestamp, ".csv")), row.names = FALSE)

  if ("abuse_num" %in% names(df_all)) {
    df_multi <- df_all[df_all$abuse_num > 1, , drop = FALSE]
    if (nrow(df_multi) > 0) {
      df_multi_prof <- _engineer_oral_health_variables(df_multi)
      prof_cols <- c("age_year", "sex", "abuse", "abuse_num", "DMFT_Index", "Care_Index", "UTN_Score", "Healthy_Rate")
      prof_cols <- prof_cols[prof_cols %in% names(df_multi_prof)]
      utils::write.csv(summary_profile(df_multi_prof[, prof_cols, drop = FALSE]), file.path(OUTPUT_DIR, paste0("multitype_profile_", timestamp, ".csv")), row.names = FALSE)
    }
  }

  # ============================================================================
  # Feature engineering (main)
  # ============================================================================
  log_info("Calculating derived variables (main)...")
  df <- _engineer_oral_health_variables(df_main)

  # ============================================================================
  # Analysis & Reporting (main)
  # ============================================================================
  log_info("Running statistical analysis (main)...")

  table1 <- create_table1_demographics(df)
  utils::write.csv(table1, file.path(OUTPUT_DIR, paste0("table1_demographics_", timestamp, ".csv")), row.names = FALSE)

  for (dent_type in c("primary_dentition", "mixed_dentition", "permanent_dentition")) {
    df_dent <- df[df$dentition_type == dent_type, , drop = FALSE]
    if (nrow(df_dent) > 0) {
      table1_dent <- create_table1_demographics(df_dent)
      utils::write.csv(table1_dent, file.path(OUTPUT_DIR, paste0("table1_demographics_", dent_type, "_", timestamp, ".csv")), row.names = FALSE)
    }
  }

  table1_1 <- create_table1_1_demographics_by_dentition(df)
  utils::write.csv(table1_1, file.path(OUTPUT_DIR, paste0("table1_1_demographics_by_dentition_", timestamp, ".csv")), row.names = FALSE)

  table2 <- create_table2_oral_health_descriptive(df)
  utils::write.csv(table2$continuous, file.path(OUTPUT_DIR, paste0("table2_continuous_", timestamp, ".csv")), row.names = FALSE)
  utils::write.csv(table2$categorical, file.path(OUTPUT_DIR, paste0("table2_categorical_", timestamp, ".csv")), row.names = FALSE)

  t3 <- create_table3_statistical_comparisons(df)
  t3_overall <- t3$overall
  t3_posthoc <- t3$posthoc
  t3_pairwise <- t3$pairwise
  t3_tidy <- t3$tidy_posthoc_pairwise
  utils::write.csv(t3_overall, file.path(OUTPUT_DIR, paste0("table3_overall_tests_", timestamp, ".csv")), row.names = FALSE)
  utils::write.csv(t3_posthoc, file.path(OUTPUT_DIR, paste0("table3_posthoc_", timestamp, ".csv")), row.names = FALSE)
  utils::write.csv(t3_pairwise, file.path(OUTPUT_DIR, paste0("table3_pairwise_mw_", timestamp, ".csv")), row.names = FALSE)

  table4_overall <- create_table4_multivariate_analysis(
    df,
    use_age_spline = TRUE,
    age_spline_df = 4,
    add_year_fe = TRUE,
    year_col = "year",
    examiner_col = examiner_col,
    id_col = subject_id_col,
    stratify_by = NULL
  )
  utils::write.csv(table4_overall, file.path(OUTPUT_DIR, paste0("table4_logistic_regression_", timestamp, ".csv")), row.names = FALSE)

  table4_dent <- create_table4_multivariate_analysis(
    df,
    use_age_spline = TRUE,
    age_spline_df = 4,
    add_year_fe = TRUE,
    year_col = "year",
    examiner_col = examiner_col,
    id_col = subject_id_col,
    stratify_by = "dentition_type",
    strata_order = c("mixed_dentition", "primary_dentition", "permanent_dentition")
  )
  if (nrow(table4_dent) > 0) {
    utils::write.csv(table4_dent, file.path(OUTPUT_DIR, paste0("table4_logistic_regression_by_dentition_", timestamp, ".csv")), row.names = FALSE)
  }

  create_forest_plot_vertical(table4_overall, df, OUTPUT_DIR, timestamp)

  table5_1 <- create_table5_1_dmft_by_dentition_abuse(df)
  utils::write.csv(table5_1$summary, file.path(OUTPUT_DIR, paste0("table5_1_dmft_by_dentition_", timestamp, ".csv")), row.names = FALSE)

  table5 <- create_table5_dmft_by_lifestage_abuse(df)
  if (nrow(table5$summary) > 0) {
    utils::write.csv(table5$summary, file.path(OUTPUT_DIR, paste0("table5_dmft_lifestage_abuse_", timestamp, ".csv")), row.names = FALSE)
  }

  table5_5 <- create_table5_5_caries_prevalence_treatment(df)
  if (nrow(table5_5$summary) > 0) {
    utils::write.csv(table5_5$summary, file.path(OUTPUT_DIR, paste0("table5_5_caries_prevalence_treatment_", timestamp, ".csv")), row.names = FALSE)
  }

  t6 <- create_table6_dmft_by_dentition_abuse(df)
  t6_summary <- t6$summary_table
  t6_within_dentition <- t6$within_dentition_posthoc
  t6_within_abuse <- t6$within_abuse_posthoc
  t6_overall_dentition <- t6$overall_dentition_posthoc

  if (nrow(t6_summary) > 0) {
    utils::write.csv(t6_summary, file.path(OUTPUT_DIR, paste0("table6_dmft_dentition_abuse_", timestamp, ".csv")), row.names = FALSE)
  }
  if (nrow(t6_within_dentition) > 0) {
    utils::write.csv(t6_within_dentition, file.path(OUTPUT_DIR, paste0("table6_within_dentition_posthoc_", timestamp, ".csv")), row.names = FALSE)
  }
  if (nrow(t6_within_abuse) > 0) {
    utils::write.csv(t6_within_abuse, file.path(OUTPUT_DIR, paste0("table6_within_abuse_posthoc_", timestamp, ".csv")), row.names = FALSE)
  }
  if (nrow(t6_overall_dentition) > 0) {
    utils::write.csv(t6_overall_dentition, file.path(OUTPUT_DIR, paste0("table6_overall_dentition_posthoc_", timestamp, ".csv")), row.names = FALSE)
  }

  plot_overall_dentition_refined(
    df = df,
    posthoc_df = t6_overall_dentition,
    y_col = "DMFT_Index",
    ylabel = "Caries Experience",
    save_path = file.path(OUTPUT_DIR, paste0("figure_overall_dentition_", timestamp, ".png"))
  )

  plot_abuse_by_dentition_facet_refined(
    df = df,
    posthoc_df = t6_within_dentition,
    y_col = "DMFT_Index",
    ylabel = "Caries Experience",
    save_path = file.path(OUTPUT_DIR, paste0("figure_abuse_by_dentition_facet_", timestamp, ".png"))
  )

  table7 <- create_table_dmft_by_year_abuse(df)
  if (nrow(table7) > 0) {
    utils::write.csv(table7, file.path(OUTPUT_DIR, paste0("table7_dmft_by_year_abuse_", timestamp, ".csv")), row.names = FALSE)
  }

  create_visualizations(df, OUTPUT_DIR)

  for (var in c("Healthy_Rate", "Baby_d", "Baby_DMFT", "Care_Index", "UTN_Score")) {
    tryCatch(
      plot_boxplot_with_dunn(df, var, group_col = "abuse", ylabel = var, output_dir = OUTPUT_DIR),
      error = function(e) message("Error drawing pairwise plot for ", var, ": ", e$message)
    )
  }

  plot_boxplot_by_dentition_type(df, output_dir = OUTPUT_DIR)
  plot_boxplot_with_dunn(df, "DMFT_Index", group_col = "abuse", ylabel = "Caries Experience", output_dir = OUTPUT_DIR)

  generate_summary_report(df, t3_overall, OUTPUT_DIR, timestamp)

  # ============================================================================
  # Sensitivity analysis: include multi-type cases and adjust for is_multitype
  # ============================================================================
  if ("abuse_num" %in% names(df_all)) {
    log_info("Running sensitivity analysis including multi-type cases...")
    df_sens <- df_all
    df_sens$is_multitype <- as.integer(df_sens$abuse_num > 1)
    df_sens <- _deduplicate_to_first_exam(df_sens, subject_id_col)

    if ("abuse" %in% names(df_sens) && is.factor(df_sens$abuse)) {
      df_sens$abuse <- droplevels(df_sens$abuse)
    }

    df_sens <- _engineer_oral_health_variables(df_sens)

    table4_sens <- create_table4_multivariate_analysis(
      df_sens,
      use_age_spline = TRUE,
      age_spline_df = 4,
      add_year_fe = TRUE,
      year_col = "year",
      examiner_col = examiner_col,
      id_col = subject_id_col,
      add_covariates = c("is_multitype")
    )

    if (nrow(table4_sens) > 0) {
      utils::write.csv(table4_sens, file.path(OUTPUT_DIR, paste0("table4_logistic_regression_sensitivity_multitype_", timestamp, ".csv")), row.names = FALSE)
    }
  }

  log_info("Analysis complete. Results saved to ", OUTPUT_DIR)
  invisible(NULL)
}

if (identical(environment(), globalenv())) {
  main()
}
