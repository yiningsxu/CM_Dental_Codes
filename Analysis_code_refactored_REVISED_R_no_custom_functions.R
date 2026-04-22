# ============================================================================
# Refactored Analysis Code (REVISED) - R procedural version
# Converted from the uploaded Python analysis scripts.
# This version avoids user-defined/custom R functions and does not source a
# separate functions file. The workflow is written as direct, line-by-line code
# blocks using base R and package functions.
# ============================================================================

# -----------------------------
# 0. Packages
# -----------------------------
required_packages <- c("dplyr", "tidyr", "ggplot2", "splines")
missing_packages <- character(0)
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    missing_packages <- c(missing_packages, pkg)
  }
}
if (length(missing_packages) > 0) {
  stop("Install these R packages before running this script: ", paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(splines))

# Optional packages:
# - PMCMRplus: Dunn post-hoc tests after Kruskal-Wallis.
# - logistf: Firth logistic regression fallback when glm has numerical problems.
has_PMCMRplus <- requireNamespace("PMCMRplus", quietly = TRUE)
has_logistf <- requireNamespace("logistf", quietly = TRUE)

timestamp <- format(Sys.Date(), "%Y%m%d")
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Starting Analysis...")

# -----------------------------
# 1. Paths and configuration
# -----------------------------
cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- cmd_args[grepl("^--file=", cmd_args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE))
} else {
  SCRIPT_DIR <- getwd()
}

BASE_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = FALSE)
DATA_DIR <- file.path(BASE_DIR, "data")
DATA_DESCRIPTION_OUTPUT_DIR <- file.path(DATA_DIR, "data_description")
OUTPUT_DIR <- file.path(BASE_DIR, "result", timestamp)
dir.create(DATA_DESCRIPTION_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
message("OUTPUT_DIR: ", OUTPUT_DIR)

ORIGINAL_DATA_NAME <- "analysisData_20260211"
ORIGINAL_DATA_PATH <- file.path(DATA_DIR, paste0(ORIGINAL_DATA_NAME, ".csv"))
END_DATE <- as.Date("2024-03-31")
target_abuse_types <- c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")
SUBJECT_ID_COL_CANDIDATES <- c("No_All", "child_id", "subject_id", "case_id", "ID", "id")
EXAMINER_COL_CANDIDATES <- c("dentist", "examiner", "doctor", "operator", "checker")

# -----------------------------
# 2. Data loading
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Loading data from ", ORIGINAL_DATA_PATH)
if (!file.exists(ORIGINAL_DATA_PATH)) {
  stop("Data file not found: ", ORIGINAL_DATA_PATH)
}

data0 <- read.csv(ORIGINAL_DATA_PATH, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Loaded data shape: ", nrow(data0), " rows x ", ncol(data0), " columns")

writeLines(names(data0), file.path(DATA_DESCRIPTION_OUTPUT_DIR, paste0(ORIGINAL_DATA_NAME, "_colnames.txt")))

# -----------------------------
# 3. Recoding and categorical order
# -----------------------------
if ("date" %in% names(data0)) {
  data0$date <- as.Date(data0$date)
}

if ("abuse" %in% names(data0)) {
  data0$abuse <- as.character(data0$abuse)
  data0$abuse[data0$abuse == "1"] <- "Physical Abuse"
  data0$abuse[data0$abuse == "2"] <- "Neglect"
  data0$abuse[data0$abuse == "3"] <- "Emotional Abuse"
  data0$abuse[data0$abuse == "4"] <- "Sexual Abuse"
  data0$abuse[data0$abuse == "5"] <- "Delinquency"
  data0$abuse[data0$abuse == "6"] <- "Parenting Difficulties"
  data0$abuse[data0$abuse == "7"] <- "Others"
  data0$abuse <- factor(data0$abuse, levels = c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse", "Delinquency", "Parenting Difficulties", "Others"), ordered = TRUE)
}

if ("occlusalRelationship" %in% names(data0)) {
  data0$occlusalRelationship <- as.character(data0$occlusalRelationship)
  data0$occlusalRelationship[data0$occlusalRelationship == "1"] <- "Normal Occlusion"
  data0$occlusalRelationship[data0$occlusalRelationship == "2"] <- "Crowding"
  data0$occlusalRelationship[data0$occlusalRelationship == "3"] <- "Anterior Crossbite"
  data0$occlusalRelationship[data0$occlusalRelationship == "4"] <- "Open Bite"
  data0$occlusalRelationship[data0$occlusalRelationship == "5"] <- "Maxillary Protrusion"
  data0$occlusalRelationship[data0$occlusalRelationship == "6"] <- "Crossbite"
  data0$occlusalRelationship[data0$occlusalRelationship == "7"] <- "Others"
  data0$occlusalRelationship <- factor(data0$occlusalRelationship, levels = c("Normal Occlusion", "Crowding", "Anterior Crossbite", "Open Bite", "Maxillary Protrusion", "Crossbite", "Others"), ordered = TRUE)
}

if ("needTOBEtreated" %in% names(data0)) {
  data0$needTOBEtreated <- as.character(data0$needTOBEtreated)
  data0$needTOBEtreated[data0$needTOBEtreated == "1"] <- "No Treatment Required"
  data0$needTOBEtreated[data0$needTOBEtreated == "2"] <- "Treatment Required"
  data0$needTOBEtreated <- factor(data0$needTOBEtreated, levels = c("No Treatment Required", "Treatment Required"), ordered = TRUE)
}

if ("emergency" %in% names(data0)) {
  data0$emergency <- as.character(data0$emergency)
  data0$emergency[data0$emergency == "1"] <- "Urgent Treatment Required"
  data0$emergency <- factor(data0$emergency, levels = c("Urgent Treatment Required"), ordered = TRUE)
}

if ("gingivitis" %in% names(data0)) {
  data0$gingivitis <- as.character(data0$gingivitis)
  data0$gingivitis[data0$gingivitis == "1"] <- "No Gingivitis"
  data0$gingivitis[data0$gingivitis == "2"] <- "Gingivitis"
  data0$gingivitis <- factor(data0$gingivitis, levels = c("No Gingivitis", "Gingivitis"), ordered = TRUE)
}

if ("OralCleanStatus" %in% names(data0)) {
  data0$OralCleanStatus <- as.character(data0$OralCleanStatus)
  data0$OralCleanStatus[data0$OralCleanStatus == "1"] <- "Poor"
  data0$OralCleanStatus[data0$OralCleanStatus == "2"] <- "Fair"
  data0$OralCleanStatus[data0$OralCleanStatus == "3"] <- "Good"
  data0$OralCleanStatus <- factor(data0$OralCleanStatus, levels = c("Poor", "Fair", "Good"), ordered = TRUE)
}

if ("habits" %in% names(data0)) {
  data0$habits <- as.character(data0$habits)
  data0$habits[data0$habits == "1"] <- "None"
  data0$habits[data0$habits == "2"] <- "Digit Sucking"
  data0$habits[data0$habits == "3"] <- "Nail biting"
  data0$habits[data0$habits == "4"] <- "Tongue Thrusting"
  data0$habits[data0$habits == "5"] <- "Smoking"
  data0$habits[data0$habits == "6"] <- "Others"
  data0$habits <- factor(data0$habits, levels = c("None", "Digit Sucking", "Nail biting", "Tongue Thrusting", "Smoking", "Others"), ordered = TRUE)
}

cleaned_path <- file.path(DATA_DIR, paste0(ORIGINAL_DATA_NAME, "_AllData_cleaned.csv"))
write.csv(data0, cleaned_path, row.names = FALSE)

# Value-count summary, written inline rather than through a helper.
exclude_cols <- c("No_All", "instruction_detail", "instruction", "memo")
value_count_rows <- list()
for (col in names(data0)) {
  if (!(col %in% exclude_cols)) {
    tab <- table(data0[[col]], useNA = "ifany")
    if (length(tab) > 0) {
      for (k in seq_along(tab)) {
        value_count_rows[[length(value_count_rows) + 1]] <- data.frame(
          Column = col,
          Value = names(tab)[k],
          Count = as.integer(tab[k]),
          stringsAsFactors = FALSE
        )
      }
    }
  }
}
if (length(value_count_rows) > 0) {
  value_counts_summary <- bind_rows(value_count_rows)
} else {
  value_counts_summary <- data.frame(Column = character(0), Value = character(0), Count = integer(0))
}
write.csv(value_counts_summary, file.path(DATA_DESCRIPTION_OUTPUT_DIR, paste0("unique_values_summary_", ORIGINAL_DATA_NAME, ".csv")), row.names = FALSE)

# Numeric descriptive profile.
numeric_cols <- names(data0)[vapply(data0, is.numeric, logical(1))]
description_rows <- list()
for (col in numeric_cols) {
  x <- data0[[col]]
  x_nonmiss <- x[!is.na(x)]
  if (length(x_nonmiss) > 0) {
    description_rows[[length(description_rows) + 1]] <- data.frame(
      Variable = col,
      N = length(x_nonmiss),
      Missing = sum(is.na(x)),
      Mean = mean(x_nonmiss),
      SD = sd(x_nonmiss),
      Min = min(x_nonmiss),
      Q1 = as.numeric(quantile(x_nonmiss, 0.25, na.rm = TRUE)),
      Median = median(x_nonmiss),
      Q3 = as.numeric(quantile(x_nonmiss, 0.75, na.rm = TRUE)),
      Max = max(x_nonmiss),
      stringsAsFactors = FALSE
    )
  }
}
if (length(description_rows) > 0) {
  data_description <- bind_rows(description_rows)
} else {
  data_description <- data.frame()
}
write.csv(data_description, file.path(DATA_DESCRIPTION_OUTPUT_DIR, paste0(ORIGINAL_DATA_NAME, "_description.csv")), row.names = FALSE)

# -----------------------------
# 4. Filtering and study-flow accounting
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Filtering data...")

df_date <- data0
if ("date" %in% names(df_date)) {
  df_date <- df_date[!is.na(df_date$date) & df_date$date <= END_DATE, , drop = FALSE]
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

subject_id_col <- NULL
for (candidate in SUBJECT_ID_COL_CANDIDATES) {
  if (candidate %in% names(df_main) && is.null(subject_id_col)) {
    subject_id_col <- candidate
  }
}

examiner_col <- NULL
for (candidate in EXAMINER_COL_CANDIDATES) {
  if (candidate %in% names(df_main) && is.null(examiner_col)) {
    examiner_col <- candidate
  }
}

if (!is.null(subject_id_col) && "date" %in% names(df_main)) {
  before_dedup <- nrow(df_main)
  df_main <- df_main[order(df_main$date), , drop = FALSE]
  df_main <- df_main[!duplicated(df_main[[subject_id_col]]), , drop = FALSE]
  after_dedup <- nrow(df_main)
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Deduplication by ", subject_id_col, ": ", before_dedup, " -> ", after_dedup, " rows (kept first exam date).")
}

if ("abuse" %in% names(df_main) && is.factor(df_main$abuse)) {
  df_main$abuse <- droplevels(df_main$abuse)
}

message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Main dataset shape: ", nrow(df_main), " rows x ", ncol(df_main), " columns")

csv_name <- paste0(ORIGINAL_DATA_NAME, "_tillMar2024_singleType_dedup")
write.csv(df_main, file.path(DATA_DIR, paste0(csv_name, ".csv")), row.names = FALSE)

flow_rows <- list()
flow_rows[[length(flow_rows) + 1]] <- data.frame(Step = "Loaded raw", N = nrow(data0), stringsAsFactors = FALSE)
flow_rows[[length(flow_rows) + 1]] <- data.frame(Step = paste0("Date <= ", format(END_DATE, "%Y-%m-%d")), N = nrow(df_date), stringsAsFactors = FALSE)
flow_rows[[length(flow_rows) + 1]] <- data.frame(Step = "Target maltreatment (abuse in 4 types) & abuse_num>=1", N = nrow(df_all), stringsAsFactors = FALSE)
if ("abuse_num" %in% names(df_all)) {
  flow_rows[[length(flow_rows) + 1]] <- data.frame(Step = "Single-type only (abuse_num==1)", N = sum(df_all$abuse_num == 1, na.rm = TRUE), stringsAsFactors = FALSE)
  flow_rows[[length(flow_rows) + 1]] <- data.frame(Step = "Multi-type excluded (abuse_num>1)", N = sum(df_all$abuse_num > 1, na.rm = TRUE), stringsAsFactors = FALSE)
}
if (!is.null(subject_id_col)) {
  flow_rows[[length(flow_rows) + 1]] <- data.frame(Step = paste0("Deduplicated to first exam per ", subject_id_col), N = nrow(df_main), stringsAsFactors = FALSE)
}
flow_summary <- bind_rows(flow_rows)
write.csv(flow_summary, file.path(OUTPUT_DIR, paste0("flow_summary_", timestamp, ".csv")), row.names = FALSE)

# -----------------------------
# 5. Feature engineering for main dataset
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Calculating derived variables (main)...")

df <- df_main

if ("age_year" %in% names(df)) {
  df$age_group <- cut(
    df$age_year,
    breaks = c(0, 6, 12, 18),
    labels = c("Early Childhood (2-6)", "Middle Childhood (7-12)", "Adolescence (13-18)"),
    right = TRUE,
    include.lowest = TRUE
  )
}

perm_teeth_cols <- c(
  paste0("U", rep(c(1, 2), each = 7), rep(1:7, times = 2)),
  paste0("L", rep(c(3, 4), each = 7), rep(1:7, times = 2))
)
baby_teeth_cols <- c(
  paste0("u", rep(c(5, 6), each = 5), rep(1:5, times = 2)),
  paste0("l", rep(c(7, 8), each = 5), rep(1:5, times = 2))
)
perm_cols <- perm_teeth_cols[perm_teeth_cols %in% names(df)]
baby_cols <- baby_teeth_cols[baby_teeth_cols %in% names(df)]
for (tc in c(perm_cols, baby_cols)) {
  df[[tc]] <- suppressWarnings(as.numeric(df[[tc]]))
}

if (length(perm_cols) > 0) {
  perm_mat <- df[, perm_cols, drop = FALSE]
  all_nan_mask_perm <- rowSums(!is.na(perm_mat)) == 0
  df$Perm_D <- rowSums(perm_mat == 3, na.rm = TRUE)
  df$Perm_M <- rowSums(perm_mat == 4, na.rm = TRUE)
  df$Perm_F <- rowSums(perm_mat == 1, na.rm = TRUE)
  df$Perm_Sound <- rowSums(perm_mat == 0, na.rm = TRUE)
  df$Perm_C0 <- rowSums(perm_mat == 2, na.rm = TRUE)
  df$Perm_total_teeth <- rowSums(!is.na(perm_mat) & perm_mat != -1, na.rm = TRUE)
  df$Perm_D[all_nan_mask_perm] <- NA_real_
  df$Perm_M[all_nan_mask_perm] <- NA_real_
  df$Perm_F[all_nan_mask_perm] <- NA_real_
  df$Perm_Sound[all_nan_mask_perm] <- NA_real_
  df$Perm_C0[all_nan_mask_perm] <- NA_real_
  df$Perm_DMFT <- df$Perm_D + df$Perm_M + df$Perm_F
  df$Perm_DMFT_C0 <- df$Perm_DMFT + df$Perm_C0
  df$Perm_sound_rate <- df$Perm_Sound / df$Perm_total_teeth * 100
  df$Perm_sound_rate[is.infinite(df$Perm_sound_rate)] <- NA_real_
} else {
  df$Perm_D <- NA_real_
  df$Perm_M <- NA_real_
  df$Perm_F <- NA_real_
  df$Perm_Sound <- NA_real_
  df$Perm_C0 <- NA_real_
  df$Perm_DMFT <- NA_real_
  df$Perm_DMFT_C0 <- NA_real_
  df$Perm_total_teeth <- 0
  df$Perm_sound_rate <- NA_real_
}

if (length(baby_cols) > 0) {
  baby_mat <- df[, baby_cols, drop = FALSE]
  all_nan_mask_baby <- rowSums(!is.na(baby_mat)) == 0
  df$Baby_d <- rowSums(baby_mat == 3, na.rm = TRUE)
  df$Baby_m <- rowSums(baby_mat == 4, na.rm = TRUE)
  df$Baby_f <- rowSums(baby_mat == 1, na.rm = TRUE)
  df$Baby_sound <- rowSums(baby_mat == 0, na.rm = TRUE)
  df$Baby_C0 <- rowSums(baby_mat == 2, na.rm = TRUE)
  df$Baby_total_teeth <- rowSums(!is.na(baby_mat) & baby_mat != -1, na.rm = TRUE)
  df$Baby_d[all_nan_mask_baby] <- NA_real_
  df$Baby_m[all_nan_mask_baby] <- NA_real_
  df$Baby_f[all_nan_mask_baby] <- NA_real_
  df$Baby_sound[all_nan_mask_baby] <- NA_real_
  df$Baby_C0[all_nan_mask_baby] <- NA_real_
  df$Baby_DMFT <- df$Baby_d + df$Baby_m + df$Baby_f
  df$Baby_DMFT_C0 <- df$Baby_DMFT + df$Baby_C0
  df$Baby_sound_rate <- df$Baby_sound / df$Baby_total_teeth * 100
  df$Baby_sound_rate[is.infinite(df$Baby_sound_rate)] <- NA_real_
} else {
  df$Baby_d <- NA_real_
  df$Baby_m <- NA_real_
  df$Baby_f <- NA_real_
  df$Baby_sound <- NA_real_
  df$Baby_C0 <- NA_real_
  df$Baby_DMFT <- NA_real_
  df$Baby_DMFT_C0 <- NA_real_
  df$Baby_total_teeth <- 0
  df$Baby_sound_rate <- NA_real_
}

both_dmft_missing <- is.na(df$Perm_DMFT) & is.na(df$Baby_DMFT)
df$DMFT_Index <- ifelse(is.na(df$Perm_DMFT), 0, df$Perm_DMFT) + ifelse(is.na(df$Baby_DMFT), 0, df$Baby_DMFT)
df$DMFT_Index[both_dmft_missing] <- NA_real_

both_dmft_c0_missing <- is.na(df$Perm_DMFT_C0) & is.na(df$Baby_DMFT_C0)
df$DMFT_C0 <- ifelse(is.na(df$Perm_DMFT_C0), 0, df$Perm_DMFT_C0) + ifelse(is.na(df$Baby_DMFT_C0), 0, df$Baby_DMFT_C0)
df$DMFT_C0[both_dmft_c0_missing] <- NA_real_

both_c0_missing <- is.na(df$Perm_C0) & is.na(df$Baby_C0)
df$C0_Count <- ifelse(is.na(df$Perm_C0), 0, df$Perm_C0) + ifelse(is.na(df$Baby_C0), 0, df$Baby_C0)
df$C0_Count[both_c0_missing] <- NA_real_

df$filled_total <- ifelse(is.na(df$Perm_F), 0, df$Perm_F) + ifelse(is.na(df$Baby_f), 0, df$Baby_f)
df$filled_total[is.na(df$Perm_F) & is.na(df$Baby_f)] <- NA_real_
df$decayed_total <- ifelse(is.na(df$Perm_D), 0, df$Perm_D) + ifelse(is.na(df$Baby_d), 0, df$Baby_d)
df$decayed_total[is.na(df$Perm_D) & is.na(df$Baby_d)] <- NA_real_
df$missing_total <- ifelse(is.na(df$Perm_M), 0, df$Perm_M) + ifelse(is.na(df$Baby_m), 0, df$Baby_m)
df$missing_total[is.na(df$Perm_M) & is.na(df$Baby_m)] <- NA_real_

df$Care_Index <- df$filled_total / df$DMFT_Index * 100
df$Care_Index[is.infinite(df$Care_Index) | df$DMFT_Index <= 0] <- NA_real_
df$UTN_Score <- df$decayed_total / df$DMFT_Index * 100
df$UTN_Score[is.infinite(df$UTN_Score) | df$DMFT_Index <= 0] <- NA_real_

df$total_teeth <- ifelse(is.na(df$Perm_total_teeth), 0, df$Perm_total_teeth) + ifelse(is.na(df$Baby_total_teeth), 0, df$Baby_total_teeth)
df$Healthy_Rate <- (ifelse(is.na(df$Perm_Sound), 0, df$Perm_Sound) + ifelse(is.na(df$Baby_sound), 0, df$Baby_sound)) / df$total_teeth * 100
df$Healthy_Rate[is.infinite(df$Healthy_Rate) | df$total_teeth <= 0] <- NA_real_

df$Present_Teeth <- df$total_teeth
df$Present_Perm_Teeth <- df$Perm_total_teeth
df$Present_Baby_Teeth <- df$Baby_total_teeth
df$has_caries <- as.integer(!is.na(df$DMFT_Index) & df$DMFT_Index > 0)
df$has_untreated_caries <- as.integer(!is.na(df$decayed_total) & df$decayed_total > 0)

present_teeth_tmp <- ifelse(is.na(df$total_teeth), 0, df$total_teeth)
present_baby_tmp <- ifelse(is.na(df$Baby_total_teeth), 0, df$Baby_total_teeth)
present_perm_tmp <- ifelse(is.na(df$Perm_total_teeth), 0, df$Perm_total_teeth)
df$dentition_type <- "mixed_dentition"
df$dentition_type[present_teeth_tmp == 0] <- "No_Teeth"
df$dentition_type[present_baby_tmp == present_teeth_tmp & present_perm_tmp == 0 & present_teeth_tmp > 0] <- "primary_dentition"
df$dentition_type[present_perm_tmp == present_teeth_tmp & present_baby_tmp == 0 & present_teeth_tmp > 0] <- "permanent_dentition"
df$dentition_type <- factor(df$dentition_type, levels = c("primary_dentition", "mixed_dentition", "permanent_dentition", "No_Teeth"))

if ("date" %in% names(df)) {
  df$year <- as.integer(format(df$date, "%Y"))
}

write.csv(df, file.path(DATA_DIR, paste0(csv_name, "_with_derived_variables.csv")), row.names = FALSE)

# Compact profile of excluded multi-type cases, using direct replicated derivation.
if ("abuse_num" %in% names(df_all)) {
  df_multi <- df_all[df_all$abuse_num > 1, , drop = FALSE]
  if (nrow(df_multi) > 0) {
    df_multi_prof <- df_multi
    if ("age_year" %in% names(df_multi_prof)) {
      df_multi_prof$age_group <- cut(df_multi_prof$age_year, breaks = c(0, 6, 12, 18), labels = c("Early Childhood (2-6)", "Middle Childhood (7-12)", "Adolescence (13-18)"), right = TRUE, include.lowest = TRUE)
    }
    perm_cols_multi <- perm_teeth_cols[perm_teeth_cols %in% names(df_multi_prof)]
    baby_cols_multi <- baby_teeth_cols[baby_teeth_cols %in% names(df_multi_prof)]
    for (tc in c(perm_cols_multi, baby_cols_multi)) {
      df_multi_prof[[tc]] <- suppressWarnings(as.numeric(df_multi_prof[[tc]]))
    }
    if (length(perm_cols_multi) > 0) {
      pm <- df_multi_prof[, perm_cols_multi, drop = FALSE]
      pm_all_na <- rowSums(!is.na(pm)) == 0
      df_multi_prof$Perm_D <- rowSums(pm == 3, na.rm = TRUE); df_multi_prof$Perm_D[pm_all_na] <- NA_real_
      df_multi_prof$Perm_M <- rowSums(pm == 4, na.rm = TRUE); df_multi_prof$Perm_M[pm_all_na] <- NA_real_
      df_multi_prof$Perm_F <- rowSums(pm == 1, na.rm = TRUE); df_multi_prof$Perm_F[pm_all_na] <- NA_real_
      df_multi_prof$Perm_Sound <- rowSums(pm == 0, na.rm = TRUE); df_multi_prof$Perm_Sound[pm_all_na] <- NA_real_
      df_multi_prof$Perm_C0 <- rowSums(pm == 2, na.rm = TRUE); df_multi_prof$Perm_C0[pm_all_na] <- NA_real_
      df_multi_prof$Perm_DMFT <- df_multi_prof$Perm_D + df_multi_prof$Perm_M + df_multi_prof$Perm_F
      df_multi_prof$Perm_total_teeth <- rowSums(!is.na(pm) & pm != -1, na.rm = TRUE)
    } else {
      df_multi_prof$Perm_D <- NA_real_; df_multi_prof$Perm_M <- NA_real_; df_multi_prof$Perm_F <- NA_real_; df_multi_prof$Perm_Sound <- NA_real_; df_multi_prof$Perm_C0 <- NA_real_; df_multi_prof$Perm_DMFT <- NA_real_; df_multi_prof$Perm_total_teeth <- 0
    }
    if (length(baby_cols_multi) > 0) {
      bm <- df_multi_prof[, baby_cols_multi, drop = FALSE]
      bm_all_na <- rowSums(!is.na(bm)) == 0
      df_multi_prof$Baby_d <- rowSums(bm == 3, na.rm = TRUE); df_multi_prof$Baby_d[bm_all_na] <- NA_real_
      df_multi_prof$Baby_m <- rowSums(bm == 4, na.rm = TRUE); df_multi_prof$Baby_m[bm_all_na] <- NA_real_
      df_multi_prof$Baby_f <- rowSums(bm == 1, na.rm = TRUE); df_multi_prof$Baby_f[bm_all_na] <- NA_real_
      df_multi_prof$Baby_sound <- rowSums(bm == 0, na.rm = TRUE); df_multi_prof$Baby_sound[bm_all_na] <- NA_real_
      df_multi_prof$Baby_C0 <- rowSums(bm == 2, na.rm = TRUE); df_multi_prof$Baby_C0[bm_all_na] <- NA_real_
      df_multi_prof$Baby_DMFT <- df_multi_prof$Baby_d + df_multi_prof$Baby_m + df_multi_prof$Baby_f
      df_multi_prof$Baby_total_teeth <- rowSums(!is.na(bm) & bm != -1, na.rm = TRUE)
    } else {
      df_multi_prof$Baby_d <- NA_real_; df_multi_prof$Baby_m <- NA_real_; df_multi_prof$Baby_f <- NA_real_; df_multi_prof$Baby_sound <- NA_real_; df_multi_prof$Baby_C0 <- NA_real_; df_multi_prof$Baby_DMFT <- NA_real_; df_multi_prof$Baby_total_teeth <- 0
    }
    df_multi_prof$DMFT_Index <- ifelse(is.na(df_multi_prof$Perm_DMFT), 0, df_multi_prof$Perm_DMFT) + ifelse(is.na(df_multi_prof$Baby_DMFT), 0, df_multi_prof$Baby_DMFT)
    df_multi_prof$DMFT_Index[is.na(df_multi_prof$Perm_DMFT) & is.na(df_multi_prof$Baby_DMFT)] <- NA_real_
    df_multi_prof$filled_total <- ifelse(is.na(df_multi_prof$Perm_F), 0, df_multi_prof$Perm_F) + ifelse(is.na(df_multi_prof$Baby_f), 0, df_multi_prof$Baby_f)
    df_multi_prof$decayed_total <- ifelse(is.na(df_multi_prof$Perm_D), 0, df_multi_prof$Perm_D) + ifelse(is.na(df_multi_prof$Baby_d), 0, df_multi_prof$Baby_d)
    df_multi_prof$Care_Index <- df_multi_prof$filled_total / df_multi_prof$DMFT_Index * 100
    df_multi_prof$Care_Index[is.infinite(df_multi_prof$Care_Index) | df_multi_prof$DMFT_Index <= 0] <- NA_real_
    df_multi_prof$UTN_Score <- df_multi_prof$decayed_total / df_multi_prof$DMFT_Index * 100
    df_multi_prof$UTN_Score[is.infinite(df_multi_prof$UTN_Score) | df_multi_prof$DMFT_Index <= 0] <- NA_real_
    df_multi_prof$total_teeth <- ifelse(is.na(df_multi_prof$Perm_total_teeth), 0, df_multi_prof$Perm_total_teeth) + ifelse(is.na(df_multi_prof$Baby_total_teeth), 0, df_multi_prof$Baby_total_teeth)
    df_multi_prof$Healthy_Rate <- (ifelse(is.na(df_multi_prof$Perm_Sound), 0, df_multi_prof$Perm_Sound) + ifelse(is.na(df_multi_prof$Baby_sound), 0, df_multi_prof$Baby_sound)) / df_multi_prof$total_teeth * 100
    df_multi_prof$Healthy_Rate[is.infinite(df_multi_prof$Healthy_Rate) | df_multi_prof$total_teeth <= 0] <- NA_real_
    prof_cols <- c("age_year", "sex", "abuse", "abuse_num", "DMFT_Index", "Care_Index", "UTN_Score", "Healthy_Rate")
    prof_cols <- prof_cols[prof_cols %in% names(df_multi_prof)]
    write.csv(summary(df_multi_prof[, prof_cols, drop = FALSE]), file.path(OUTPUT_DIR, paste0("multitype_profile_", timestamp, ".csv")))
  }
}

# -----------------------------
# 6. Table 1: demographics overall and by dentition
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Creating Table 1...")

table1_sources <- list()
table1_sources[["overall"]] <- df
for (dent_type in c("primary_dentition", "mixed_dentition", "permanent_dentition")) {
  if ("dentition_type" %in% names(df)) {
    df_dent <- df[df$dentition_type == dent_type, , drop = FALSE]
    if (nrow(df_dent) > 0) {
      table1_sources[[dent_type]] <- df_dent
    }
  }
}

for (source_name in names(table1_sources)) {
  df_table <- table1_sources[[source_name]]
  if (is.factor(df_table$abuse)) {
    abuse_types <- levels(droplevels(df_table$abuse))
  } else {
    abuse_types <- sort(unique(as.character(df_table$abuse[!is.na(df_table$abuse)])))
  }

  table1_rows <- list()
  total_row <- data.frame(Variable = "Total N", Category = "", stringsAsFactors = FALSE)
  for (abuse in abuse_types) {
    total_row[[abuse]] <- as.character(sum(df_table$abuse == abuse, na.rm = TRUE))
  }
  total_row$Total <- as.character(nrow(df_table))
  total_row$`p-value` <- ""
  table1_rows[[length(table1_rows) + 1]] <- total_row

  if ("sex" %in% names(df_table)) {
    sex_header <- data.frame(Variable = "Sex", Category = "", stringsAsFactors = FALSE)
    for (abuse in abuse_types) sex_header[[abuse]] <- ""
    sex_header$Total <- ""
    sex_header$`p-value` <- ""
    table1_rows[[length(table1_rows) + 1]] <- sex_header

    p_sex <- NA_real_
    sex_tab <- table(df_table$abuse, df_table$sex)
    if (nrow(sex_tab) >= 2 && ncol(sex_tab) >= 2) {
      sex_test <- try(chisq.test(sex_tab), silent = TRUE)
      if (!inherits(sex_test, "try-error")) p_sex <- sex_test$p.value
    }
    sex_values <- c("Male", "Female")
    sex_values <- c(sex_values[sex_values %in% unique(as.character(df_table$sex))], sort(setdiff(unique(as.character(df_table$sex[!is.na(df_table$sex)])), sex_values)))
    first_sex <- TRUE
    for (sex_value in sex_values) {
      row <- data.frame(Variable = "", Category = paste0("  ", sex_value), stringsAsFactors = FALSE)
      for (abuse in abuse_types) {
        n_cell <- sum(df_table$abuse == abuse & df_table$sex == sex_value, na.rm = TRUE)
        n_group <- sum(df_table$abuse == abuse & !is.na(df_table$sex), na.rm = TRUE)
        pct <- ifelse(n_group > 0, n_cell / n_group * 100, 0)
        row[[abuse]] <- sprintf("%d (%.1f%%)", n_cell, pct)
      }
      total_n <- sum(df_table$sex == sex_value, na.rm = TRUE)
      total_pct <- ifelse(nrow(df_table) > 0, total_n / nrow(df_table) * 100, 0)
      row$Total <- sprintf("%d (%.1f%%)", total_n, total_pct)
      row$`p-value` <- ifelse(first_sex & !is.na(p_sex), sprintf("%.3f", p_sex), "")
      first_sex <- FALSE
      table1_rows[[length(table1_rows) + 1]] <- row
    }
  }

  if ("age_year" %in% names(df_table)) {
    age_row <- data.frame(Variable = "Age (years)", Category = "Mean ± SD", stringsAsFactors = FALSE)
    for (abuse in abuse_types) {
      x <- df_table$age_year[df_table$abuse == abuse]
      x <- x[!is.na(x)]
      age_row[[abuse]] <- ifelse(length(x) > 0, sprintf("%.1f ± %.1f", mean(x), sd(x)), "N/A")
    }
    x_total <- df_table$age_year[!is.na(df_table$age_year)]
    age_row$Total <- ifelse(length(x_total) > 0, sprintf("%.1f ± %.1f", mean(x_total), sd(x_total)), "N/A")
    p_age <- NA_real_
    age_kw_data <- df_table[!is.na(df_table$age_year) & !is.na(df_table$abuse), , drop = FALSE]
    if (length(unique(age_kw_data$abuse)) >= 2) {
      age_kw <- try(kruskal.test(age_year ~ abuse, data = age_kw_data), silent = TRUE)
      if (!inherits(age_kw, "try-error")) p_age <- age_kw$p.value
    }
    age_row$`p-value` <- ifelse(!is.na(p_age), sprintf("%.3f", p_age), "N/A")
    table1_rows[[length(table1_rows) + 1]] <- age_row

    age_median_row <- data.frame(Variable = "", Category = "Median [IQR]", stringsAsFactors = FALSE)
    for (abuse in abuse_types) {
      x <- df_table$age_year[df_table$abuse == abuse]
      x <- x[!is.na(x)]
      if (length(x) > 0) {
        q <- quantile(x, c(0.25, 0.5, 0.75), na.rm = TRUE)
        age_median_row[[abuse]] <- sprintf("%.0f [%.0f-%.0f]", q[2], q[1], q[3])
      } else {
        age_median_row[[abuse]] <- "N/A"
      }
    }
    if (length(x_total) > 0) {
      q <- quantile(x_total, c(0.25, 0.5, 0.75), na.rm = TRUE)
      age_median_row$Total <- sprintf("%.0f [%.0f-%.0f]", q[2], q[1], q[3])
    } else {
      age_median_row$Total <- "N/A"
    }
    age_median_row$`p-value` <- ""
    table1_rows[[length(table1_rows) + 1]] <- age_median_row
  }

  if ("age_group" %in% names(df_table)) {
    age_group_header <- data.frame(Variable = "Age Group", Category = "", stringsAsFactors = FALSE)
    for (abuse in abuse_types) age_group_header[[abuse]] <- ""
    age_group_header$Total <- ""
    age_group_header$`p-value` <- ""
    table1_rows[[length(table1_rows) + 1]] <- age_group_header

    p_age_grp <- NA_real_
    age_group_valid <- df_table[!is.na(df_table$age_group) & !is.na(df_table$abuse), , drop = FALSE]
    if (nrow(age_group_valid) > 0) {
      age_group_tab <- table(age_group_valid$abuse, age_group_valid$age_group)
      if (nrow(age_group_tab) >= 2 && ncol(age_group_tab) >= 2) {
        age_group_test <- try(chisq.test(age_group_tab), silent = TRUE)
        if (!inherits(age_group_test, "try-error")) p_age_grp <- age_group_test$p.value
      }
    }
    if (is.factor(df_table$age_group)) {
      age_group_values <- levels(droplevels(df_table$age_group))
    } else {
      age_group_values <- sort(unique(as.character(df_table$age_group[!is.na(df_table$age_group)])))
    }
    first_age_group <- TRUE
    for (age_group in age_group_values) {
      row <- data.frame(Variable = "", Category = paste0("  ", age_group), stringsAsFactors = FALSE)
      for (abuse in abuse_types) {
        n_cell <- sum(df_table$abuse == abuse & df_table$age_group == age_group, na.rm = TRUE)
        n_group <- sum(df_table$abuse == abuse & !is.na(df_table$age_group), na.rm = TRUE)
        pct <- ifelse(n_group > 0, n_cell / n_group * 100, 0)
        row[[abuse]] <- sprintf("%d (%.1f%%)", n_cell, pct)
      }
      total_n <- sum(df_table$age_group == age_group, na.rm = TRUE)
      total_valid <- sum(!is.na(df_table$age_group))
      total_pct <- ifelse(total_valid > 0, total_n / total_valid * 100, 0)
      row$Total <- sprintf("%d (%.1f%%)", total_n, total_pct)
      row$`p-value` <- ifelse(first_age_group & !is.na(p_age_grp), sprintf("%.3f", p_age_grp), "")
      first_age_group <- FALSE
      table1_rows[[length(table1_rows) + 1]] <- row
    }
  }

  table1 <- bind_rows(table1_rows)
  if (source_name == "overall") {
    write.csv(table1, file.path(OUTPUT_DIR, paste0("table1_demographics_", timestamp, ".csv")), row.names = FALSE)
  } else {
    write.csv(table1, file.path(OUTPUT_DIR, paste0("table1_demographics_", source_name, "_", timestamp, ".csv")), row.names = FALSE)
  }
}

# -----------------------------
# 7. Table 1.1: demographics by dentition and abuse type
# -----------------------------
dentition_order <- c("primary_dentition", "mixed_dentition", "permanent_dentition")
abuse_types <- if (is.factor(df$abuse)) levels(droplevels(df$abuse)) else sort(unique(as.character(df$abuse[!is.na(df$abuse)])))
table1_1_rows <- list()
for (dent_type in dentition_order) {
  df_dent <- df[df$dentition_type == dent_type, , drop = FALSE]
  if (nrow(df_dent) == 0 || !("age_year" %in% names(df_dent))) next
  age_total <- df_dent$age_year[!is.na(df_dent$age_year)]
  if (length(age_total) > 0) {
    table1_1_rows[[length(table1_1_rows) + 1]] <- data.frame(
      Dentition_Period = dent_type,
      Group = "Total",
      N = length(age_total),
      Mean = round(mean(age_total), 2),
      SD = round(sd(age_total), 2),
      Median = round(median(age_total), 2),
      IQR = sprintf("%.2f-%.2f", quantile(age_total, 0.25), quantile(age_total, 0.75)),
      Min = round(min(age_total), 2),
      Max = round(max(age_total), 2),
      `Mean±SD` = sprintf("%.2f ± %.2f", mean(age_total), sd(age_total)),
      `Median[IQR]` = sprintf("%.1f [%.1f-%.1f]", median(age_total), quantile(age_total, 0.25), quantile(age_total, 0.75)),
      `Min-Max` = sprintf("%.1f-%.1f", min(age_total), max(age_total)),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }
  for (abuse in abuse_types) {
    age_sub <- df_dent$age_year[df_dent$abuse == abuse]
    age_sub <- age_sub[!is.na(age_sub)]
    if (length(age_sub) > 0) {
      table1_1_rows[[length(table1_1_rows) + 1]] <- data.frame(
        Dentition_Period = dent_type,
        Group = abuse,
        N = length(age_sub),
        Mean = round(mean(age_sub), 2),
        SD = round(sd(age_sub), 2),
        Median = round(median(age_sub), 2),
        IQR = sprintf("%.2f-%.2f", quantile(age_sub, 0.25), quantile(age_sub, 0.75)),
        Min = round(min(age_sub), 2),
        Max = round(max(age_sub), 2),
        `Mean±SD` = sprintf("%.2f ± %.2f", mean(age_sub), sd(age_sub)),
        `Median[IQR]` = sprintf("%.1f [%.1f-%.1f]", median(age_sub), quantile(age_sub, 0.25), quantile(age_sub, 0.75)),
        `Min-Max` = sprintf("%.1f-%.1f", min(age_sub), max(age_sub)),
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
    }
  }
}
table1_1 <- if (length(table1_1_rows) > 0) bind_rows(table1_1_rows) else data.frame()
write.csv(table1_1, file.path(OUTPUT_DIR, paste0("table1_1_demographics_by_dentition_", timestamp, ".csv")), row.names = FALSE)

# -----------------------------
# 8. Table 2: oral-health descriptive statistics
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Creating Table 2...")

continuous_vars <- data.frame(
  var = c("DMFT_Index", "decayed_total", "missing_total", "filled_total", "Perm_DMFT", "Baby_DMFT", "Perm_D", "Perm_M", "Perm_F", "Baby_d", "Baby_m", "Baby_f", "C0_Count", "Healthy_Rate", "Care_Index", "UTN_Score", "Trauma_Count", "RDT_Count"),
  label = c("DMFT Index (Total)", "Decayed Total (D+d)", "Missing Total (M+m)", "Filled Total (F+f)", "Permanent DMFT", "Primary dmft", "Permanent D (Decayed)", "Permanent M (Missing)", "Permanent F (Filled)", "Primary d (decayed)", "Primary m (missing)", "Primary f (filled)", "C0 (Incipient Caries)", "Healthy Teeth Rate (%)", "Care Index (%) (DMFT_Index>0 only)", "Untreated Caries Rate (%) (DMFT_Index>0 only)", "Dental Trauma Count", "Retained Deciduous Teeth"),
  stringsAsFactors = FALSE
)
ratio_vars <- c("Care_Index", "UTN_Score")
table2_cont_rows <- list()
for (i in seq_len(nrow(continuous_vars))) {
  var_name <- continuous_vars$var[i]
  var_label <- continuous_vars$label[i]
  if (!(var_name %in% names(df))) next
  df_var_all <- df
  if (var_name %in% ratio_vars && "DMFT_Index" %in% names(df_var_all)) df_var_all <- df_var_all[df_var_all$DMFT_Index > 0, , drop = FALSE]
  row <- data.frame(Variable = var_label, stringsAsFactors = FALSE)
  for (abuse in abuse_types) {
    subset <- df_var_all[[var_name]][df_var_all$abuse == abuse]
    subset <- subset[!is.na(subset)]
    if (length(subset) > 0) {
      row[[paste0(abuse, "_Mean_SD")]] <- sprintf("%.2f ± %.2f", mean(subset), sd(subset))
      row[[paste0(abuse, "_Median_IQR")]] <- sprintf("%.1f [%.1f-%.1f]", median(subset), quantile(subset, 0.25), quantile(subset, 0.75))
    } else {
      row[[paste0(abuse, "_Mean_SD")]] <- "N/A"
      row[[paste0(abuse, "_Median_IQR")]] <- "N/A"
    }
  }
  total <- df_var_all[[var_name]][!is.na(df_var_all[[var_name]])]
  if (length(total) > 0) {
    row$Total_Mean_SD <- sprintf("%.2f ± %.2f", mean(total), sd(total))
    row$Total_Median_IQR <- sprintf("%.1f [%.1f-%.1f]", median(total), quantile(total, 0.25), quantile(total, 0.75))
  } else {
    row$Total_Mean_SD <- "N/A"
    row$Total_Median_IQR <- "N/A"
  }
  kw_data <- df_var_all[!is.na(df_var_all[[var_name]]) & !is.na(df_var_all$abuse), , drop = FALSE]
  p_val <- NA_real_
  if (length(unique(kw_data$abuse)) >= 2) {
    kw_test <- try(kruskal.test(kw_data[[var_name]] ~ kw_data$abuse), silent = TRUE)
    if (!inherits(kw_test, "try-error")) p_val <- kw_test$p.value
  }
  row$`p-value` <- ifelse(is.na(p_val), "N/A", ifelse(p_val < 0.0001, "<0.0001", sprintf("%.4f", p_val)))
  table2_cont_rows[[length(table2_cont_rows) + 1]] <- row
}
table2_cont <- if (length(table2_cont_rows) > 0) bind_rows(table2_cont_rows) else data.frame()
write.csv(table2_cont, file.path(OUTPUT_DIR, paste0("table2_continuous_", timestamp, ".csv")), row.names = FALSE)

categorical_vars <- data.frame(
  var = c("gingivitis", "needTOBEtreated", "occlusalRelationship", "OralCleanStatus", "habits"),
  label = c("Gingivitis", "Treatment Need", "Occlusal Relationship", "Oral Hygiene Status", "Oral Habits"),
  stringsAsFactors = FALSE
)
table2_cat_rows <- list()
for (i in seq_len(nrow(categorical_vars))) {
  var_name <- categorical_vars$var[i]
  var_label <- categorical_vars$label[i]
  if (!(var_name %in% names(df))) next
  header_row <- data.frame(Variable = var_label, Category = "", stringsAsFactors = FALSE)
  for (abuse in abuse_types) {
    header_row[[paste0(abuse, "_n")]] <- ""
    header_row[[paste0(abuse, "_%")]] <- ""
  }
  header_row$Total_n <- ""
  header_row$Total_pct <- ""
  header_row$`p-value` <- ""
  table2_cat_rows[[length(table2_cat_rows) + 1]] <- header_row
  df_valid <- df[!is.na(df[[var_name]]) & !is.na(df$abuse), , drop = FALSE]
  p_val <- NA_real_
  if (nrow(df_valid) > 0) {
    tab <- table(df_valid$abuse, df_valid[[var_name]])
    if (nrow(tab) >= 2 && ncol(tab) >= 2) {
      chi_test <- try(chisq.test(tab), silent = TRUE)
      if (!inherits(chi_test, "try-error")) p_val <- chi_test$p.value
    }
  }
  if (is.factor(df[[var_name]])) {
    categories <- levels(droplevels(df[[var_name]]))
  } else {
    categories <- sort(unique(as.character(df[[var_name]][!is.na(df[[var_name]])])))
  }
  first_cat <- TRUE
  for (cat in categories) {
    row <- data.frame(Variable = "", Category = paste0("  ", cat), stringsAsFactors = FALSE)
    for (abuse in abuse_types) {
      n_cell <- sum(df$abuse == abuse & df[[var_name]] == cat, na.rm = TRUE)
      n_group <- sum(df$abuse == abuse & !is.na(df[[var_name]]), na.rm = TRUE)
      pct <- ifelse(n_group > 0, n_cell / n_group * 100, 0)
      row[[paste0(abuse, "_n")]] <- as.character(n_cell)
      row[[paste0(abuse, "_%")]] <- sprintf("%.1f", pct)
    }
    total_n <- sum(df[[var_name]] == cat, na.rm = TRUE)
    total_valid <- sum(!is.na(df[[var_name]]))
    total_pct <- ifelse(total_valid > 0, total_n / total_valid * 100, 0)
    row$Total_n <- as.character(total_n)
    row$Total_pct <- sprintf("%.1f", total_pct)
    row$`p-value` <- ifelse(first_cat & !is.na(p_val), ifelse(p_val < 0.0001, "<0.0001", sprintf("%.4f", p_val)), "")
    first_cat <- FALSE
    table2_cat_rows[[length(table2_cat_rows) + 1]] <- row
  }
}
table2_cat <- if (length(table2_cat_rows) > 0) bind_rows(table2_cat_rows) else data.frame()
write.csv(table2_cat, file.path(OUTPUT_DIR, paste0("table2_categorical_", timestamp, ".csv")), row.names = FALSE)

# -----------------------------
# 9. Table 3: Kruskal-Wallis and post-hoc tests
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Creating Table 3...")

table3_vars <- c("DMFT_Index", "decayed_total", "missing_total", "filled_total", "Perm_DMFT", "Baby_DMFT", "Perm_D", "Perm_M", "Perm_F", "Baby_d", "Baby_m", "Baby_f", "C0_Count", "Healthy_Rate", "Care_Index", "UTN_Score", "Trauma_Count", "DMFT_C0", "Perm_DMFT_C0", "Baby_DMFT_C0")
table3_vars <- table3_vars[table3_vars %in% names(df)]
t3_overall_rows <- list()
t3_posthoc_rows <- list()
t3_pairwise_rows <- list()
t3_tidy_rows <- list()

for (var_name in table3_vars) {
  df_var <- df
  if (var_name %in% ratio_vars && "DMFT_Index" %in% names(df_var)) df_var <- df_var[df_var$DMFT_Index > 0, , drop = FALSE]
  df_var <- df_var[!is.na(df_var[[var_name]]) & !is.na(df_var$abuse), , drop = FALSE]
  if (length(unique(df_var$abuse)) < 2) next

  row <- data.frame(Variable = var_name, Test = "Kruskal-Wallis", stringsAsFactors = FALSE)
  total_data <- df_var[[var_name]][!is.na(df_var[[var_name]])]
  if (length(total_data) > 0) {
    row$Total_Mean_SD <- sprintf("%.2f ± %.2f", mean(total_data), sd(total_data))
    row$Total_Median_IQR <- sprintf("%.1f [%.1f-%.1f]", median(total_data), quantile(total_data, 0.25), quantile(total_data, 0.75))
  } else {
    row$Total_Mean_SD <- "N/A"
    row$Total_Median_IQR <- "N/A"
  }
  for (abuse in abuse_types) {
    x <- df_var[[var_name]][df_var$abuse == abuse]
    x <- x[!is.na(x)]
    if (length(x) > 0) {
      row[[paste0(abuse, "_Mean_SD")]] <- sprintf("%.2f ± %.2f", mean(x), sd(x))
      row[[paste0(abuse, "_Median_IQR")]] <- sprintf("%.1f [%.1f-%.1f]", median(x), quantile(x, 0.25), quantile(x, 0.75))
    } else {
      row[[paste0(abuse, "_Mean_SD")]] <- "N/A"
      row[[paste0(abuse, "_Median_IQR")]] <- "N/A"
    }
  }
  kw_test <- try(kruskal.test(df_var[[var_name]] ~ df_var$abuse), silent = TRUE)
  p_kw <- NA_real_
  h_stat <- NA_real_
  if (!inherits(kw_test, "try-error")) {
    p_kw <- kw_test$p.value
    h_stat <- as.numeric(kw_test$statistic)
  }
  row$Statistic <- ifelse(is.na(h_stat), "N/A", sprintf("%.3f", h_stat))
  row$`p-value` <- ifelse(is.na(p_kw), "N/A", ifelse(p_kw < 0.0001, "<0.0001", sprintf("%.4f", p_kw)))
  row$Significant <- ifelse(!is.na(p_kw) & p_kw < 0.05, "Yes", "No")
  t3_overall_rows[[length(t3_overall_rows) + 1]] <- row

  # Post-hoc after significant KW.
  if (!is.na(p_kw) && p_kw < 0.05) {
    df_var$rank_value <- rank(df_var[[var_name]], ties.method = "average")
    mean_rank_table <- aggregate(rank_value ~ abuse, data = df_var, FUN = mean)

    p_adj_matrix <- NULL
    p_unadj_matrix <- NULL
    posthoc_label <- "Dunn (PMCMRplus)"
    if (has_PMCMRplus) {
      dunn_adj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_var[[var_name]], g = df_var$abuse, p.adjust.method = "bonferroni"), silent = TRUE)
      dunn_unadj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_var[[var_name]], g = df_var$abuse, p.adjust.method = "none"), silent = TRUE)
      if (!inherits(dunn_adj, "try-error") && !inherits(dunn_unadj, "try-error")) {
        p_adj_matrix <- dunn_adj$p.value
        p_unadj_matrix <- dunn_unadj$p.value
      }
    }
    if (is.null(p_adj_matrix)) {
      posthoc_label <- "Pairwise Wilcoxon fallback"
      pw_adj <- try(pairwise.wilcox.test(df_var[[var_name]], df_var$abuse, p.adjust.method = "bonferroni", exact = FALSE), silent = TRUE)
      pw_unadj <- try(pairwise.wilcox.test(df_var[[var_name]], df_var$abuse, p.adjust.method = "none", exact = FALSE), silent = TRUE)
      if (!inherits(pw_adj, "try-error") && !inherits(pw_unadj, "try-error")) {
        p_adj_matrix <- pw_adj$p.value
        p_unadj_matrix <- pw_unadj$p.value
      }
    }

    if (!is.null(p_adj_matrix)) {
      for (i_abuse in seq_len(length(abuse_types) - 1)) {
        for (j_abuse in seq((i_abuse + 1), length(abuse_types))) {
          abuse1 <- abuse_types[i_abuse]
          abuse2 <- abuse_types[j_abuse]
          p_adj <- NA_real_
          p_unadj <- NA_real_
          if (abuse1 %in% rownames(p_adj_matrix) && abuse2 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[abuse1, abuse2]
          if (abuse2 %in% rownames(p_adj_matrix) && abuse1 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[abuse2, abuse1]
          if (abuse1 %in% rownames(p_unadj_matrix) && abuse2 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[abuse1, abuse2]
          if (abuse2 %in% rownames(p_unadj_matrix) && abuse1 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[abuse2, abuse1]
          if (is.na(p_adj)) next

          vals1 <- df_var[[var_name]][df_var$abuse == abuse1]
          vals2 <- df_var[[var_name]][df_var$abuse == abuse2]
          vals1 <- vals1[!is.na(vals1)]
          vals2 <- vals2[!is.na(vals2)]
          if (length(vals1) == 0 || length(vals2) == 0) next
          mr1 <- mean_rank_table$rank_value[mean_rank_table$abuse == abuse1]
          mr2 <- mean_rank_table$rank_value[mean_rank_table$abuse == abuse2]
          q1_vals1 <- quantile(vals1, c(0.25, 0.75), na.rm = TRUE)
          q1_vals2 <- quantile(vals2, c(0.25, 0.75), na.rm = TRUE)

          t3_posthoc_rows[[length(t3_posthoc_rows) + 1]] <- data.frame(
            Variable = var_name,
            Group1 = abuse1,
            Group2 = abuse2,
            Comparison = paste0(abuse1, " vs ", abuse2),
            Group1_n = length(vals1),
            Group2_n = length(vals2),
            Group1_Mean = round(mean(vals1), 2),
            Group2_Mean = round(mean(vals2), 2),
            Group1_SD = round(sd(vals1), 2),
            Group2_SD = round(sd(vals2), 2),
            Group1_Median = round(median(vals1), 2),
            Group2_Median = round(median(vals2), 2),
            Group1_IQR = sprintf("%.1f-%.1f", q1_vals1[1], q1_vals1[2]),
            Group2_IQR = sprintf("%.1f-%.1f", q1_vals2[1], q1_vals2[2]),
            Group1_Mean_SD = sprintf("%.2f ± %.2f", mean(vals1), sd(vals1)),
            Group2_Mean_SD = sprintf("%.2f ± %.2f", mean(vals2), sd(vals2)),
            Group1_Median_IQR = sprintf("%.1f [%.1f-%.1f]", median(vals1), q1_vals1[1], q1_vals1[2]),
            Group2_Median_IQR = sprintf("%.1f [%.1f-%.1f]", median(vals2), q1_vals2[1], q1_vals2[2]),
            Group1_Mean_Rank = round(mr1, 2),
            Group2_Mean_Rank = round(mr2, 2),
            `p-value (unadjusted)` = ifelse(is.na(p_unadj), "N/A", ifelse(p_unadj < 0.0001, "<0.0001", sprintf("%.4f", p_unadj))),
            `p-value (adjusted)` = ifelse(is.na(p_adj), "N/A", ifelse(p_adj < 0.0001, "<0.0001", sprintf("%.4f", p_adj))),
            Significant = ifelse(!is.na(p_adj) & p_adj < 0.05, "Yes", "No"),
            Method = posthoc_label,
            check.names = FALSE,
            stringsAsFactors = FALSE
          )

          t3_tidy_rows[[length(t3_tidy_rows) + 1]] <- data.frame(
            variable = var_name,
            group1 = abuse1,
            group2 = abuse2,
            group1_n = length(vals1),
            group2_n = length(vals2),
            group1_mean = mean(vals1),
            group2_mean = mean(vals2),
            group1_sd = sd(vals1),
            group2_sd = sd(vals2),
            group1_median = median(vals1),
            group2_median = median(vals2),
            group1_q1 = q1_vals1[1],
            group1_q3 = q1_vals1[2],
            group2_q1 = q1_vals2[1],
            group2_q3 = q1_vals2[2],
            group1_mean_rank = mr1,
            group2_mean_rank = mr2,
            p_unadjusted = p_unadj,
            p_adjusted = p_adj,
            significant = !is.na(p_adj) & p_adj < 0.05,
            analysis_type = "Table 3: Overall",
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }

  # Pairwise Mann-Whitney sensitivity.
  n_pairs <- choose(length(abuse_types), 2)
  bonf_threshold <- ifelse(n_pairs * length(table3_vars) > 0, 0.05 / (n_pairs * length(table3_vars)), 0.05)
  for (i_abuse in seq_len(length(abuse_types) - 1)) {
    for (j_abuse in seq((i_abuse + 1), length(abuse_types))) {
      abuse1 <- abuse_types[i_abuse]
      abuse2 <- abuse_types[j_abuse]
      vals1 <- df_var[[var_name]][df_var$abuse == abuse1]
      vals2 <- df_var[[var_name]][df_var$abuse == abuse2]
      vals1 <- vals1[!is.na(vals1)]
      vals2 <- vals2[!is.na(vals2)]
      if (length(vals1) == 0 || length(vals2) == 0) next
      mw <- try(wilcox.test(vals1, vals2, alternative = "two.sided", exact = FALSE), silent = TRUE)
      if (inherits(mw, "try-error")) next
      u_stat <- as.numeric(mw$statistic)
      p_val <- mw$p.value
      r_val <- 1 - (2 * u_stat) / (length(vals1) * length(vals2))
      t3_pairwise_rows[[length(t3_pairwise_rows) + 1]] <- data.frame(
        Variable = var_name,
        Group1 = abuse1,
        Group2 = abuse2,
        Group1_Median = sprintf("%.1f", median(vals1)),
        Group2_Median = sprintf("%.1f", median(vals2)),
        U_Statistic = sprintf("%.0f", u_stat),
        `p-value` = ifelse(p_val < 0.0001, "<0.0001", sprintf("%.4f", p_val)),
        Effect_Size_r = sprintf("%.3f", r_val),
        Significant_Bonferroni = ifelse(p_val < bonf_threshold, "Yes", "No"),
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
    }
  }
}

t3_overall <- if (length(t3_overall_rows) > 0) bind_rows(t3_overall_rows) else data.frame()
t3_posthoc <- if (length(t3_posthoc_rows) > 0) bind_rows(t3_posthoc_rows) else data.frame()
t3_pairwise <- if (length(t3_pairwise_rows) > 0) bind_rows(t3_pairwise_rows) else data.frame()
t3_tidy <- if (length(t3_tidy_rows) > 0) bind_rows(t3_tidy_rows) else data.frame()
write.csv(t3_overall, file.path(OUTPUT_DIR, paste0("table3_overall_tests_", timestamp, ".csv")), row.names = FALSE)
write.csv(t3_posthoc, file.path(OUTPUT_DIR, paste0("table3_posthoc_", timestamp, ".csv")), row.names = FALSE)
write.csv(t3_pairwise, file.path(OUTPUT_DIR, paste0("table3_pairwise_mw_", timestamp, ".csv")), row.names = FALSE)
if (nrow(t3_tidy) > 0) write.csv(t3_tidy, file.path(OUTPUT_DIR, paste0("table3_tidy_posthoc_", timestamp, ".csv")), row.names = FALSE)

# -----------------------------
# 10. Table 4: multivariable logistic regression
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Creating Table 4...")

if ("sex" %in% names(df)) {
  df$sex_male <- as.integer(df$sex == "Male")
}
if ("gingivitis" %in% names(df)) {
  df$gingivitis_binary <- as.integer(df$gingivitis == "Gingivitis")
}
if ("needTOBEtreated" %in% names(df)) {
  df$treatment_need <- as.integer(df$needTOBEtreated == "Treatment Required")
}

outcome_vars <- c("has_caries", "has_untreated_caries")
outcome_labels <- c("Caries Experience (>0)", "Untreated Caries")
if ("gingivitis_binary" %in% names(df)) {
  outcome_vars <- c(outcome_vars, "gingivitis_binary")
  outcome_labels <- c(outcome_labels, "Gingivitis")
}
if ("treatment_need" %in% names(df)) {
  outcome_vars <- c(outcome_vars, "treatment_need")
  outcome_labels <- c(outcome_labels, "Treatment Need")
}

reference_category <- "Physical Abuse"
comparison_categories <- c("Neglect", "Emotional Abuse", "Sexual Abuse")
strata_values <- c("Overall", dentition_order)
table4_rows <- list()

for (stratum_label in strata_values) {
  if (stratum_label == "Overall") {
    df_stratum <- df
  } else {
    df_stratum <- df[df$dentition_type == stratum_label, , drop = FALSE]
  }
  if (nrow(df_stratum) == 0) next
  for (out_i in seq_along(outcome_vars)) {
    outcome_var <- outcome_vars[out_i]
    outcome_label <- outcome_labels[out_i]
    if (!(outcome_var %in% names(df_stratum))) next
    for (comparison in comparison_categories) {
      df_model <- df_stratum[df_stratum$abuse %in% c(reference_category, comparison), , drop = FALSE]
      if (!("age_year" %in% names(df_model)) || !("sex_male" %in% names(df_model))) next
      df_model$comparison <- as.integer(df_model$abuse == comparison)
      needed_cols <- c(outcome_var, "age_year", "sex_male", "comparison", "abuse")
      if ("year" %in% names(df_model)) needed_cols <- c(needed_cols, "year")
      if (!is.null(examiner_col) && examiner_col %in% names(df_model)) needed_cols <- c(needed_cols, examiner_col)
      if (!is.null(subject_id_col) && subject_id_col %in% names(df_model)) needed_cols <- c(needed_cols, subject_id_col)
      needed_cols <- unique(needed_cols[needed_cols %in% names(df_model)])
      df_model <- df_model[, needed_cols, drop = FALSE]
      df_model <- df_model[complete.cases(df_model[, c(outcome_var, "age_year", "sex_male", "comparison"), drop = FALSE]), , drop = FALSE]
      if (nrow(df_model) < 50) next
      if (length(unique(df_model[[outcome_var]])) < 2) next

      rhs_terms <- c("splines::ns(age_year, df = 4)", "sex_male", "comparison")
      adjusted_for <- c("Age (spline)", "Sex")
      if ("year" %in% names(df_model)) {
        rhs_terms <- c(rhs_terms, "factor(year)")
        adjusted_for <- c(adjusted_for, "Year (FE)")
      }
      if (!is.null(examiner_col) && examiner_col %in% names(df_model)) {
        rhs_terms <- c(rhs_terms, paste0("factor(", examiner_col, ")"))
        adjusted_for <- c(adjusted_for, "Examiner (FE)")
      }
      model_formula <- as.formula(paste(outcome_var, "~", paste(rhs_terms, collapse = " + ")))
      fit <- try(glm(model_formula, data = df_model, family = binomial()), silent = TRUE)
      model_name <- "Logit (glm)"
      beta <- NA_real_
      se <- NA_real_
      p_val <- NA_real_
      if (!inherits(fit, "try-error")) {
        coefs <- summary(fit)$coefficients
        if ("comparison" %in% rownames(coefs)) {
          beta <- coefs["comparison", "Estimate"]
          se <- coefs["comparison", "Std. Error"]
          p_val <- coefs["comparison", "Pr(>|z|)"]
        }
      }
      if ((is.na(beta) || is.na(se) || !is.finite(beta) || !is.finite(se)) && has_logistf) {
        fit_firth <- try(logistf::logistf(model_formula, data = df_model), silent = TRUE)
        if (!inherits(fit_firth, "try-error")) {
          beta <- fit_firth$coefficients["comparison"]
          se <- sqrt(diag(fit_firth$var))["comparison"]
          p_val <- fit_firth$prob["comparison"]
          model_name <- "Logit (Firth/logistf)"
        }
      }
      or_val <- exp(beta)
      ci_low <- exp(beta - 1.96 * se)
      ci_up <- exp(beta + 1.96 * se)
      table4_rows[[length(table4_rows) + 1]] <- data.frame(
        Stratum = ifelse(stratum_label == "Overall", "", stratum_label),
        Outcome = outcome_label,
        Comparison = paste0(comparison, " vs ", reference_category),
        N = nrow(df_model),
        Events = sum(df_model[[outcome_var]], na.rm = TRUE),
        `Odds Ratio` = ifelse(is.finite(or_val), sprintf("%.2f", or_val), "N/A"),
        `95% CI` = ifelse(is.finite(ci_low) & is.finite(ci_up), sprintf("(%.2f-%.2f)", ci_low, ci_up), "N/A"),
        `p-value` = ifelse(is.na(p_val), "N/A", ifelse(p_val < 0.0001, "<0.0001", sprintf("%.4f", p_val))),
        Model = model_name,
        Adjusted_for = paste(adjusted_for, collapse = ", "),
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
    }
  }
}

table4_all <- if (length(table4_rows) > 0) bind_rows(table4_rows) else data.frame()
table4_overall <- table4_all[table4_all$Stratum == "", , drop = FALSE]
table4_dent <- table4_all[table4_all$Stratum != "", , drop = FALSE]
write.csv(table4_overall, file.path(OUTPUT_DIR, paste0("table4_logistic_regression_", timestamp, ".csv")), row.names = FALSE)
if (nrow(table4_dent) > 0) write.csv(table4_dent, file.path(OUTPUT_DIR, paste0("table4_logistic_regression_by_dentition_", timestamp, ".csv")), row.names = FALSE)

# Forest plot for overall Table 4.
if (nrow(table4_overall) > 0) {
  forest_df <- table4_overall[table4_overall$`Odds Ratio` != "N/A" & table4_overall$`95% CI` != "N/A", , drop = FALSE]
  if (nrow(forest_df) > 0) {
    forest_df$OR <- suppressWarnings(as.numeric(forest_df$`Odds Ratio`))
    forest_df$CI_lower <- suppressWarnings(as.numeric(gsub("^\\(([^-]+)-.*$", "\\1", forest_df$`95% CI`)))
    forest_df$CI_upper <- suppressWarnings(as.numeric(gsub("^\\([^-]+-([^)]+)\\)$", "\\1", forest_df$`95% CI`)))
    forest_df$Plot_Label <- paste(forest_df$Outcome, forest_df$Comparison, sep = " | ")
    forest_df <- forest_df[!is.na(forest_df$OR) & !is.na(forest_df$CI_lower) & !is.na(forest_df$CI_upper), , drop = FALSE]
    if (nrow(forest_df) > 0) {
      p_forest <- ggplot(forest_df, aes(x = OR, y = reorder(Plot_Label, OR))) +
        geom_vline(xintercept = 1, linetype = "dashed") +
        geom_segment(aes(x = CI_lower, xend = CI_upper, y = reorder(Plot_Label, OR), yend = reorder(Plot_Label, OR))) +
        geom_point(size = 2) +
        labs(x = "Odds Ratio (95% CI)", y = NULL) +
        theme_minimal()
      ggsave(file.path(OUTPUT_DIR, paste0("figure_forest_plot_", timestamp, ".png")), p_forest, width = 10, height = 8, dpi = 300)
    }
  }
}

# -----------------------------
# 11. Table 5.1 and Table 6: DMFT by dentition and abuse type
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Creating Tables 5.1 and 6...")

table6_summary_rows <- list()
t6_within_dentition_rows <- list()
for (dent_type in dentition_order) {
  df_dent <- df[df$dentition_type == dent_type & !is.na(df$DMFT_Index) & !is.na(df$abuse), , drop = FALSE]
  if (nrow(df_dent) == 0) next
  p_kw <- NA_real_
  if (length(unique(df_dent$abuse)) >= 2) {
    kw <- try(kruskal.test(DMFT_Index ~ abuse, data = df_dent), silent = TRUE)
    if (!inherits(kw, "try-error")) p_kw <- kw$p.value
  }
  overall_subset <- df_dent$DMFT_Index[!is.na(df_dent$DMFT_Index)]
  if (length(overall_subset) > 0) {
    table6_summary_rows[[length(table6_summary_rows) + 1]] <- data.frame(
      Dentition_Type = dent_type,
      Abuse_Type = "Total",
      N = length(overall_subset),
      Mean = round(mean(overall_subset), 2),
      SD = round(sd(overall_subset), 2),
      Median = round(median(overall_subset), 2),
      IQR = sprintf("%.2f-%.2f", quantile(overall_subset, 0.25), quantile(overall_subset, 0.75)),
      Min = round(min(overall_subset), 2),
      Max = round(max(overall_subset), 2),
      Mean_SD = sprintf("%.2f ± %.2f", mean(overall_subset), sd(overall_subset)),
      Median_IQR = sprintf("%.1f [%.1f-%.1f]", median(overall_subset), quantile(overall_subset, 0.25), quantile(overall_subset, 0.75)),
      `Min-Max` = sprintf("%.1f-%.1f", min(overall_subset), max(overall_subset)),
      `p-value (KW within dentition)` = ifelse(is.na(p_kw), "N/A", ifelse(p_kw < 0.0001, "<0.0001", sprintf("%.4f", p_kw))),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
  }
  first_row <- TRUE
  for (abuse in abuse_types) {
    subset <- df_dent$DMFT_Index[df_dent$abuse == abuse]
    subset <- subset[!is.na(subset)]
    if (length(subset) == 0) next
    table6_summary_rows[[length(table6_summary_rows) + 1]] <- data.frame(
      Dentition_Type = ifelse(first_row, dent_type, ""),
      Abuse_Type = abuse,
      N = length(subset),
      Mean = round(mean(subset), 2),
      SD = round(sd(subset), 2),
      Median = round(median(subset), 2),
      IQR = sprintf("%.2f-%.2f", quantile(subset, 0.25), quantile(subset, 0.75)),
      Min = round(min(subset), 2),
      Max = round(max(subset), 2),
      Mean_SD = sprintf("%.2f ± %.2f", mean(subset), sd(subset)),
      Median_IQR = sprintf("%.1f [%.1f-%.1f]", median(subset), quantile(subset, 0.25), quantile(subset, 0.75)),
      `Min-Max` = sprintf("%.1f-%.1f", min(subset), max(subset)),
      `p-value (KW within dentition)` = ifelse(first_row, ifelse(is.na(p_kw), "N/A", ifelse(p_kw < 0.0001, "<0.0001", sprintf("%.4f", p_kw))), ""),
      check.names = FALSE,
      stringsAsFactors = FALSE
    )
    first_row <- FALSE
  }

  if (!is.na(p_kw) && p_kw < 0.05) {
    df_dent$rank_value <- rank(df_dent$DMFT_Index, ties.method = "average")
    mean_rank_table <- aggregate(rank_value ~ abuse, data = df_dent, FUN = mean)
    p_adj_matrix <- NULL
    p_unadj_matrix <- NULL
    posthoc_label <- "Dunn (PMCMRplus)"
    if (has_PMCMRplus) {
      dunn_adj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_dent$DMFT_Index, g = df_dent$abuse, p.adjust.method = "bonferroni"), silent = TRUE)
      dunn_unadj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_dent$DMFT_Index, g = df_dent$abuse, p.adjust.method = "none"), silent = TRUE)
      if (!inherits(dunn_adj, "try-error") && !inherits(dunn_unadj, "try-error")) {
        p_adj_matrix <- dunn_adj$p.value
        p_unadj_matrix <- dunn_unadj$p.value
      }
    }
    if (is.null(p_adj_matrix)) {
      posthoc_label <- "Pairwise Wilcoxon fallback"
      pw_adj <- try(pairwise.wilcox.test(df_dent$DMFT_Index, df_dent$abuse, p.adjust.method = "bonferroni", exact = FALSE), silent = TRUE)
      pw_unadj <- try(pairwise.wilcox.test(df_dent$DMFT_Index, df_dent$abuse, p.adjust.method = "none", exact = FALSE), silent = TRUE)
      if (!inherits(pw_adj, "try-error") && !inherits(pw_unadj, "try-error")) {
        p_adj_matrix <- pw_adj$p.value
        p_unadj_matrix <- pw_unadj$p.value
      }
    }
    if (!is.null(p_adj_matrix)) {
      for (i_abuse in seq_len(length(abuse_types) - 1)) {
        for (j_abuse in seq((i_abuse + 1), length(abuse_types))) {
          abuse1 <- abuse_types[i_abuse]; abuse2 <- abuse_types[j_abuse]
          p_adj <- NA_real_; p_unadj <- NA_real_
          if (abuse1 %in% rownames(p_adj_matrix) && abuse2 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[abuse1, abuse2]
          if (abuse2 %in% rownames(p_adj_matrix) && abuse1 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[abuse2, abuse1]
          if (abuse1 %in% rownames(p_unadj_matrix) && abuse2 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[abuse1, abuse2]
          if (abuse2 %in% rownames(p_unadj_matrix) && abuse1 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[abuse2, abuse1]
          if (is.na(p_adj)) next
          vals1 <- df_dent$DMFT_Index[df_dent$abuse == abuse1]
          vals2 <- df_dent$DMFT_Index[df_dent$abuse == abuse2]
          vals1 <- vals1[!is.na(vals1)]; vals2 <- vals2[!is.na(vals2)]
          if (length(vals1) == 0 || length(vals2) == 0) next
          q1 <- quantile(vals1, c(0.25, 0.75)); q2 <- quantile(vals2, c(0.25, 0.75))
          mr1 <- mean_rank_table$rank_value[mean_rank_table$abuse == abuse1]
          mr2 <- mean_rank_table$rank_value[mean_rank_table$abuse == abuse2]
          t6_within_dentition_rows[[length(t6_within_dentition_rows) + 1]] <- data.frame(
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
            Group1_SD = round(sd(vals1), 2),
            Group2_SD = round(sd(vals2), 2),
            Group1_Median = round(median(vals1), 2),
            Group2_Median = round(median(vals2), 2),
            Group1_IQR = sprintf("%.2f-%.2f", q1[1], q1[2]),
            Group2_IQR = sprintf("%.2f-%.2f", q2[1], q2[2]),
            Group1_Mean_SD = sprintf("%.2f ± %.2f", mean(vals1), sd(vals1)),
            Group2_Mean_SD = sprintf("%.2f ± %.2f", mean(vals2), sd(vals2)),
            Group1_Median_IQR = sprintf("%.2f [%.2f-%.2f]", median(vals1), q1[1], q1[2]),
            Group2_Median_IQR = sprintf("%.2f [%.2f-%.2f]", median(vals2), q2[1], q2[2]),
            Group1_Mean_Rank = round(mr1, 2),
            Group2_Mean_Rank = round(mr2, 2),
            KW_p_value = ifelse(p_kw < 0.0001, "<0.0001", sprintf("%.4f", p_kw)),
            `p-value (unadjusted)` = ifelse(is.na(p_unadj), "N/A", ifelse(p_unadj < 0.0001, "<0.0001", sprintf("%.4f", p_unadj))),
            `p-value (adjusted)` = ifelse(is.na(p_adj), "N/A", ifelse(p_adj < 0.0001, "<0.0001", sprintf("%.4f", p_adj))),
            Significant = ifelse(!is.na(p_adj) & p_adj < 0.05, "Yes", "No"),
            Method = posthoc_label,
            check.names = FALSE,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }
}

t6_summary <- if (length(table6_summary_rows) > 0) bind_rows(table6_summary_rows) else data.frame()
t6_within_dentition <- if (length(t6_within_dentition_rows) > 0) bind_rows(t6_within_dentition_rows) else data.frame()
write.csv(t6_summary, file.path(OUTPUT_DIR, paste0("table6_dmft_dentition_abuse_", timestamp, ".csv")), row.names = FALSE)
if (nrow(t6_within_dentition) > 0) write.csv(t6_within_dentition, file.path(OUTPUT_DIR, paste0("table6_within_dentition_posthoc_", timestamp, ".csv")), row.names = FALSE)

# Table 5.1 uses a compact copy of the Table 6 dentition x abuse summary.
table5_1 <- t6_summary
write.csv(table5_1, file.path(OUTPUT_DIR, paste0("table5_1_dmft_by_dentition_", timestamp, ".csv")), row.names = FALSE)

# Within each abuse subtype: compare dentition types.
t6_within_abuse_rows <- list()
for (abuse in abuse_types) {
  df_abuse <- df[df$abuse == abuse & !is.na(df$DMFT_Index) & !is.na(df$dentition_type), , drop = FALSE]
  df_abuse <- df_abuse[df_abuse$dentition_type %in% dentition_order, , drop = FALSE]
  if (length(unique(df_abuse$dentition_type)) < 2) next
  kw <- try(kruskal.test(DMFT_Index ~ dentition_type, data = df_abuse), silent = TRUE)
  if (inherits(kw, "try-error")) next
  p_kw <- kw$p.value
  if (is.na(p_kw) || p_kw >= 0.05) next
  df_abuse$rank_value <- rank(df_abuse$DMFT_Index, ties.method = "average")
  mean_rank_table <- aggregate(rank_value ~ dentition_type, data = df_abuse, FUN = mean)
  p_adj_matrix <- NULL; p_unadj_matrix <- NULL; posthoc_label <- "Dunn (PMCMRplus)"
  if (has_PMCMRplus) {
    dunn_adj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_abuse$DMFT_Index, g = df_abuse$dentition_type, p.adjust.method = "bonferroni"), silent = TRUE)
    dunn_unadj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_abuse$DMFT_Index, g = df_abuse$dentition_type, p.adjust.method = "none"), silent = TRUE)
    if (!inherits(dunn_adj, "try-error") && !inherits(dunn_unadj, "try-error")) { p_adj_matrix <- dunn_adj$p.value; p_unadj_matrix <- dunn_unadj$p.value }
  }
  if (is.null(p_adj_matrix)) {
    posthoc_label <- "Pairwise Wilcoxon fallback"
    pw_adj <- try(pairwise.wilcox.test(df_abuse$DMFT_Index, df_abuse$dentition_type, p.adjust.method = "bonferroni", exact = FALSE), silent = TRUE)
    pw_unadj <- try(pairwise.wilcox.test(df_abuse$DMFT_Index, df_abuse$dentition_type, p.adjust.method = "none", exact = FALSE), silent = TRUE)
    if (!inherits(pw_adj, "try-error") && !inherits(pw_unadj, "try-error")) { p_adj_matrix <- pw_adj$p.value; p_unadj_matrix <- pw_unadj$p.value }
  }
  if (is.null(p_adj_matrix)) next
  for (i_dent in seq_len(length(dentition_order) - 1)) {
    for (j_dent in seq((i_dent + 1), length(dentition_order))) {
      dent1 <- dentition_order[i_dent]; dent2 <- dentition_order[j_dent]
      p_adj <- NA_real_; p_unadj <- NA_real_
      if (dent1 %in% rownames(p_adj_matrix) && dent2 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[dent1, dent2]
      if (dent2 %in% rownames(p_adj_matrix) && dent1 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[dent2, dent1]
      if (dent1 %in% rownames(p_unadj_matrix) && dent2 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[dent1, dent2]
      if (dent2 %in% rownames(p_unadj_matrix) && dent1 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[dent2, dent1]
      if (is.na(p_adj)) next
      vals1 <- df_abuse$DMFT_Index[df_abuse$dentition_type == dent1]
      vals2 <- df_abuse$DMFT_Index[df_abuse$dentition_type == dent2]
      vals1 <- vals1[!is.na(vals1)]; vals2 <- vals2[!is.na(vals2)]
      if (length(vals1) == 0 || length(vals2) == 0) next
      q1 <- quantile(vals1, c(0.25, 0.75)); q2 <- quantile(vals2, c(0.25, 0.75))
      mr1 <- mean_rank_table$rank_value[mean_rank_table$dentition_type == dent1]
      mr2 <- mean_rank_table$rank_value[mean_rank_table$dentition_type == dent2]
      t6_within_abuse_rows[[length(t6_within_abuse_rows) + 1]] <- data.frame(
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
        Group1_SD = round(sd(vals1), 2),
        Group2_SD = round(sd(vals2), 2),
        Group1_Median = round(median(vals1), 2),
        Group2_Median = round(median(vals2), 2),
        Group1_IQR = sprintf("%.2f-%.2f", q1[1], q1[2]),
        Group2_IQR = sprintf("%.2f-%.2f", q2[1], q2[2]),
        Group1_Mean_Rank = round(mr1, 2),
        Group2_Mean_Rank = round(mr2, 2),
        KW_p_value = ifelse(p_kw < 0.0001, "<0.0001", sprintf("%.4f", p_kw)),
        `p-value (unadjusted)` = ifelse(is.na(p_unadj), "N/A", ifelse(p_unadj < 0.0001, "<0.0001", sprintf("%.4f", p_unadj))),
        `p-value (adjusted)` = ifelse(is.na(p_adj), "N/A", ifelse(p_adj < 0.0001, "<0.0001", sprintf("%.4f", p_adj))),
        Significant = ifelse(!is.na(p_adj) & p_adj < 0.05, "Yes", "No"),
        Method = posthoc_label,
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
    }
  }
}
t6_within_abuse <- if (length(t6_within_abuse_rows) > 0) bind_rows(t6_within_abuse_rows) else data.frame()
if (nrow(t6_within_abuse) > 0) write.csv(t6_within_abuse, file.path(OUTPUT_DIR, paste0("table6_within_abuse_posthoc_", timestamp, ".csv")), row.names = FALSE)

# Overall dentition comparison.
t6_overall_dentition_rows <- list()
df_overall_dent <- df[!is.na(df$DMFT_Index) & df$dentition_type %in% dentition_order, , drop = FALSE]
if (length(unique(df_overall_dent$dentition_type)) >= 2) {
  kw <- try(kruskal.test(DMFT_Index ~ dentition_type, data = df_overall_dent), silent = TRUE)
  if (!inherits(kw, "try-error")) {
    p_kw <- kw$p.value
    if (!is.na(p_kw) && p_kw < 0.05) {
      df_overall_dent$rank_value <- rank(df_overall_dent$DMFT_Index, ties.method = "average")
      mean_rank_table <- aggregate(rank_value ~ dentition_type, data = df_overall_dent, FUN = mean)
      p_adj_matrix <- NULL; p_unadj_matrix <- NULL; posthoc_label <- "Dunn (PMCMRplus)"
      if (has_PMCMRplus) {
        dunn_adj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_overall_dent$DMFT_Index, g = df_overall_dent$dentition_type, p.adjust.method = "bonferroni"), silent = TRUE)
        dunn_unadj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_overall_dent$DMFT_Index, g = df_overall_dent$dentition_type, p.adjust.method = "none"), silent = TRUE)
        if (!inherits(dunn_adj, "try-error") && !inherits(dunn_unadj, "try-error")) { p_adj_matrix <- dunn_adj$p.value; p_unadj_matrix <- dunn_unadj$p.value }
      }
      if (is.null(p_adj_matrix)) {
        posthoc_label <- "Pairwise Wilcoxon fallback"
        pw_adj <- try(pairwise.wilcox.test(df_overall_dent$DMFT_Index, df_overall_dent$dentition_type, p.adjust.method = "bonferroni", exact = FALSE), silent = TRUE)
        pw_unadj <- try(pairwise.wilcox.test(df_overall_dent$DMFT_Index, df_overall_dent$dentition_type, p.adjust.method = "none", exact = FALSE), silent = TRUE)
        if (!inherits(pw_adj, "try-error") && !inherits(pw_unadj, "try-error")) { p_adj_matrix <- pw_adj$p.value; p_unadj_matrix <- pw_unadj$p.value }
      }
      if (!is.null(p_adj_matrix)) {
        for (i_dent in seq_len(length(dentition_order) - 1)) {
          for (j_dent in seq((i_dent + 1), length(dentition_order))) {
            dent1 <- dentition_order[i_dent]; dent2 <- dentition_order[j_dent]
            p_adj <- NA_real_; p_unadj <- NA_real_
            if (dent1 %in% rownames(p_adj_matrix) && dent2 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[dent1, dent2]
            if (dent2 %in% rownames(p_adj_matrix) && dent1 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[dent2, dent1]
            if (dent1 %in% rownames(p_unadj_matrix) && dent2 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[dent1, dent2]
            if (dent2 %in% rownames(p_unadj_matrix) && dent1 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[dent2, dent1]
            if (is.na(p_adj)) next
            vals1 <- df_overall_dent$DMFT_Index[df_overall_dent$dentition_type == dent1]
            vals2 <- df_overall_dent$DMFT_Index[df_overall_dent$dentition_type == dent2]
            vals1 <- vals1[!is.na(vals1)]; vals2 <- vals2[!is.na(vals2)]
            if (length(vals1) == 0 || length(vals2) == 0) next
            q1 <- quantile(vals1, c(0.25, 0.75)); q2 <- quantile(vals2, c(0.25, 0.75))
            mr1 <- mean_rank_table$rank_value[mean_rank_table$dentition_type == dent1]
            mr2 <- mean_rank_table$rank_value[mean_rank_table$dentition_type == dent2]
            t6_overall_dentition_rows[[length(t6_overall_dentition_rows) + 1]] <- data.frame(
              Analysis = "Overall dentition comparison",
              Variable = "DMFT_Index",
              Group1 = dent1,
              Group2 = dent2,
              Comparison = paste0(dent1, " vs ", dent2),
              Group1_n = length(vals1),
              Group2_n = length(vals2),
              Group1_Mean = round(mean(vals1), 2),
              Group2_Mean = round(mean(vals2), 2),
              Group1_SD = round(sd(vals1), 2),
              Group2_SD = round(sd(vals2), 2),
              Group1_Median = round(median(vals1), 2),
              Group2_Median = round(median(vals2), 2),
              Group1_IQR = sprintf("%.2f-%.2f", q1[1], q1[2]),
              Group2_IQR = sprintf("%.2f-%.2f", q2[1], q2[2]),
              Group1_Mean_Rank = round(mr1, 2),
              Group2_Mean_Rank = round(mr2, 2),
              KW_p_value = ifelse(p_kw < 0.0001, "<0.0001", sprintf("%.4f", p_kw)),
              `p-value (unadjusted)` = ifelse(is.na(p_unadj), "N/A", ifelse(p_unadj < 0.0001, "<0.0001", sprintf("%.4f", p_unadj))),
              `p-value (adjusted)` = ifelse(is.na(p_adj), "N/A", ifelse(p_adj < 0.0001, "<0.0001", sprintf("%.4f", p_adj))),
              Significant = ifelse(!is.na(p_adj) & p_adj < 0.05, "Yes", "No"),
              Method = posthoc_label,
              check.names = FALSE,
              stringsAsFactors = FALSE
            )
          }
        }
      }
    }
  }
}
t6_overall_dentition <- if (length(t6_overall_dentition_rows) > 0) bind_rows(t6_overall_dentition_rows) else data.frame()
if (nrow(t6_overall_dentition) > 0) write.csv(t6_overall_dentition, file.path(OUTPUT_DIR, paste0("table6_overall_dentition_posthoc_", timestamp, ".csv")), row.names = FALSE)

# -----------------------------
# 12. Table 5: DMFT by life stage and abuse type
# -----------------------------
table5_rows <- list()
life_stage_order <- c("Early Childhood (2-6)", "Middle Childhood (7-12)", "Adolescence (13-18)")
if ("age_group" %in% names(df)) {
  life_stages <- c(life_stage_order[life_stage_order %in% unique(as.character(df$age_group))], setdiff(sort(unique(as.character(df$age_group[!is.na(df$age_group)]))), life_stage_order))
  for (life_stage in life_stages) {
    df_stage <- df[df$age_group == life_stage & !is.na(df$DMFT_Index), , drop = FALSE]
    if (nrow(df_stage) == 0) next
    p_kw <- NA_real_
    if (length(unique(df_stage$abuse)) >= 2) {
      kw <- try(kruskal.test(DMFT_Index ~ abuse, data = df_stage), silent = TRUE)
      if (!inherits(kw, "try-error")) p_kw <- kw$p.value
    }
    first_row <- TRUE
    for (abuse in abuse_types) {
      subset <- df_stage$DMFT_Index[df_stage$abuse == abuse]
      subset <- subset[!is.na(subset)]
      if (length(subset) == 0) next
      table5_rows[[length(table5_rows) + 1]] <- data.frame(
        Life_Stage = ifelse(first_row, life_stage, ""),
        Abuse_Type = abuse,
        N = length(subset),
        Mean = sprintf("%.2f", mean(subset)),
        SD = sprintf("%.2f", sd(subset)),
        Median = sprintf("%.1f", median(subset)),
        `25%` = sprintf("%.1f", quantile(subset, 0.25)),
        `75%` = sprintf("%.1f", quantile(subset, 0.75)),
        Min = sprintf("%.0f", min(subset)),
        Max = sprintf("%.0f", max(subset)),
        `p-value (KW)` = ifelse(first_row, ifelse(is.na(p_kw), "N/A", ifelse(p_kw < 0.0001, "<0.0001", sprintf("%.4f", p_kw))), ""),
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
      first_row <- FALSE
    }
  }
}
table5 <- if (length(table5_rows) > 0) bind_rows(table5_rows) else data.frame()
if (nrow(table5) > 0) write.csv(table5, file.path(OUTPUT_DIR, paste0("table5_dmft_lifestage_abuse_", timestamp, ".csv")), row.names = FALSE)

# -----------------------------
# 13. Table 5.5: caries prevalence and treatment status
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Creating Table 5.5...")

table5_5_rows <- list()
header <- data.frame(Variable = "=== CARIES PREVALENCE ===", Category = "", stringsAsFactors = FALSE)
for (abuse in abuse_types) header[[abuse]] <- ""
header$Total <- ""
header$`p-value` <- ""
table5_5_rows[[length(table5_5_rows) + 1]] <- header

prevalence_labels <- c("Children with Caries", "Untreated Caries (Decayed)", "Missing Teeth (Missing)", "Filled Teeth (Filled)")
prevalence_cols <- c("DMFT_Index", "decayed_total", "missing_total", "filled_total")
prevalence_defs <- c("DMFT_Index > 0", "decayed_total > 0", "missing_total > 0", "filled_total > 0")
for (i in seq_along(prevalence_cols)) {
  var_col <- prevalence_cols[i]
  if (!(var_col %in% names(df))) next
  row <- data.frame(Variable = prevalence_labels[i], Category = prevalence_defs[i], stringsAsFactors = FALSE)
  for (abuse in abuse_types) {
    subset <- df[df$abuse == abuse, , drop = FALSE]
    n_total <- nrow(subset)
    n_prev <- sum(subset[[var_col]] > 0, na.rm = TRUE)
    pct <- ifelse(n_total > 0, n_prev / n_total * 100, 0)
    row[[abuse]] <- sprintf("%d/%d (%.1f%%)", n_prev, n_total, pct)
  }
  n_total_all <- nrow(df)
  n_prev_all <- sum(df[[var_col]] > 0, na.rm = TRUE)
  pct_all <- ifelse(n_total_all > 0, n_prev_all / n_total_all * 100, 0)
  row$Total <- sprintf("%d/%d (%.1f%%)", n_prev_all, n_total_all, pct_all)
  binary_col <- as.integer(df[[var_col]] > 0)
  p_val <- NA_real_
  tab <- table(df$abuse, binary_col)
  if (nrow(tab) >= 2 && ncol(tab) >= 2) {
    chi <- try(chisq.test(tab), silent = TRUE)
    if (!inherits(chi, "try-error")) p_val <- chi$p.value
  }
  row$`p-value` <- ifelse(is.na(p_val), "N/A", ifelse(p_val < 0.0001, "<0.0001", sprintf("%.4f", p_val)))
  table5_5_rows[[length(table5_5_rows) + 1]] <- row
}

header <- data.frame(Variable = "=== TREATMENT STATUS ===", Category = "", stringsAsFactors = FALSE)
for (abuse in abuse_types) header[[abuse]] <- ""
header$Total <- ""
header$`p-value` <- ""
table5_5_rows[[length(table5_5_rows) + 1]] <- header

df_with_caries <- df[df$DMFT_Index > 0, , drop = FALSE]
if (nrow(df_with_caries) > 0) {
  row <- data.frame(Variable = "Fully Treated Caries", Category = "f+F = DMFT_Index", stringsAsFactors = FALSE)
  for (abuse in abuse_types) {
    subset <- df_with_caries[df_with_caries$abuse == abuse, , drop = FALSE]
    n_total <- nrow(subset)
    n_fully <- sum(subset$filled_total == subset$DMFT_Index, na.rm = TRUE)
    pct <- ifelse(n_total > 0, n_fully / n_total * 100, 0)
    row[[abuse]] <- sprintf("%d/%d (%.1f%%)", n_fully, n_total, pct)
  }
  n_fully_all <- sum(df_with_caries$filled_total == df_with_caries$DMFT_Index, na.rm = TRUE)
  row$Total <- sprintf("%d/%d (%.1f%%)", n_fully_all, nrow(df_with_caries), n_fully_all / nrow(df_with_caries) * 100)
  is_fully <- as.integer(df_with_caries$filled_total == df_with_caries$DMFT_Index)
  tab <- table(df_with_caries$abuse, is_fully)
  p_val <- NA_real_
  if (nrow(tab) >= 2 && ncol(tab) >= 2) {
    chi <- try(chisq.test(tab), silent = TRUE)
    if (!inherits(chi, "try-error")) p_val <- chi$p.value
  }
  row$`p-value` <- ifelse(is.na(p_val), "N/A", ifelse(p_val < 0.0001, "<0.0001", sprintf("%.4f", p_val)))
  table5_5_rows[[length(table5_5_rows) + 1]] <- row

  row <- data.frame(Variable = "No Filled Teeth", Category = "f+F = 0 (Among Caries Active)", stringsAsFactors = FALSE)
  for (abuse in abuse_types) {
    subset <- df_with_caries[df_with_caries$abuse == abuse, , drop = FALSE]
    n_total <- nrow(subset)
    n_no_filled <- sum(subset$filled_total == 0, na.rm = TRUE)
    pct <- ifelse(n_total > 0, n_no_filled / n_total * 100, 0)
    row[[abuse]] <- sprintf("%d/%d (%.1f%%)", n_no_filled, n_total, pct)
  }
  n_no_filled_all <- sum(df_with_caries$filled_total == 0, na.rm = TRUE)
  row$Total <- sprintf("%d/%d (%.1f%%)", n_no_filled_all, nrow(df_with_caries), n_no_filled_all / nrow(df_with_caries) * 100)
  has_no_filled <- as.integer(df$DMFT_Index > 0 & df$filled_total == 0)
  tab <- table(df$abuse, has_no_filled)
  p_val <- NA_real_
  if (nrow(tab) >= 2 && ncol(tab) >= 2) {
    chi <- try(chisq.test(tab), silent = TRUE)
    if (!inherits(chi, "try-error")) p_val <- chi$p.value
  }
  row$`p-value` <- ifelse(is.na(p_val), "N/A", ifelse(p_val < 0.0001, "<0.0001", sprintf("%.4f", p_val)))
  table5_5_rows[[length(table5_5_rows) + 1]] <- row
}

header <- data.frame(Variable = "=== DMFT WITH C0 ===", Category = "", stringsAsFactors = FALSE)
for (abuse in abuse_types) header[[abuse]] <- ""
header$Total <- ""
header$`p-value` <- ""
table5_5_rows[[length(table5_5_rows) + 1]] <- header

c0_vars <- c("DMFT_C0", "Perm_DMFT_C0", "Baby_DMFT_C0")
c0_labels <- c("Total DMFT + C0", "Permanent DMFT + C0", "Primary dmft + C0")
for (i in seq_along(c0_vars)) {
  var_name <- c0_vars[i]
  if (!(var_name %in% names(df))) next
  row <- data.frame(Variable = c0_labels[i], Category = "Mean ± SD", stringsAsFactors = FALSE)
  for (abuse in abuse_types) {
    subset <- df[[var_name]][df$abuse == abuse]
    subset <- subset[!is.na(subset)]
    row[[abuse]] <- ifelse(length(subset) > 0, sprintf("%.2f ± %.2f", mean(subset), sd(subset)), "N/A")
  }
  total <- df[[var_name]][!is.na(df[[var_name]])]
  row$Total <- ifelse(length(total) > 0, sprintf("%.2f ± %.2f", mean(total), sd(total)), "N/A")
  p_val <- NA_real_
  kw_data <- df[!is.na(df[[var_name]]) & !is.na(df$abuse), , drop = FALSE]
  if (length(unique(kw_data$abuse)) >= 2) {
    kw <- try(kruskal.test(kw_data[[var_name]] ~ kw_data$abuse), silent = TRUE)
    if (!inherits(kw, "try-error")) p_val <- kw$p.value
  }
  row$`p-value` <- ifelse(is.na(p_val), "N/A", ifelse(p_val < 0.0001, "<0.0001", sprintf("%.4f", p_val)))
  table5_5_rows[[length(table5_5_rows) + 1]] <- row

  row <- data.frame(Variable = "", Category = "Median [IQR]", stringsAsFactors = FALSE)
  for (abuse in abuse_types) {
    subset <- df[[var_name]][df$abuse == abuse]
    subset <- subset[!is.na(subset)]
    row[[abuse]] <- ifelse(length(subset) > 0, sprintf("%.1f [%.1f-%.1f]", median(subset), quantile(subset, 0.25), quantile(subset, 0.75)), "N/A")
  }
  row$Total <- ifelse(length(total) > 0, sprintf("%.1f [%.1f-%.1f]", median(total), quantile(total, 0.25), quantile(total, 0.75)), "N/A")
  row$`p-value` <- ""
  table5_5_rows[[length(table5_5_rows) + 1]] <- row
}

table5_5 <- if (length(table5_5_rows) > 0) bind_rows(table5_5_rows) else data.frame()
write.csv(table5_5, file.path(OUTPUT_DIR, paste0("table5_5_caries_prevalence_treatment_", timestamp, ".csv")), row.names = FALSE)

# -----------------------------
# 14. Table 7: DMFT, Dt, Mt, Ft by year and abuse type
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Creating Table 7...")

table7_rows <- list()
if ("year" %in% names(df)) {
  df$Dt <- ifelse(is.na(df$Perm_D), 0, df$Perm_D) + ifelse(is.na(df$Baby_d), 0, df$Baby_d)
  df$Mt <- ifelse(is.na(df$Perm_M), 0, df$Perm_M) + ifelse(is.na(df$Baby_m), 0, df$Baby_m)
  df$Ft <- ifelse(is.na(df$Perm_F), 0, df$Perm_F) + ifelse(is.na(df$Baby_f), 0, df$Baby_f)
  df$DFt <- df$Dt + df$Ft
  years <- sort(unique(df$year[!is.na(df$year)]))
  vars_to_summarize <- c("DMFT_Index", "Perm_DMFT", "Baby_DMFT", "Dt", "Mt", "Ft", "DFt")
  names_to_summarize <- c("DMFT", "Perm_DMFT", "Baby_DMFT", "Dt (Untreated)", "Mt (Missing)", "Ft (Filled)", "DFt (Dt+Ft)")
  for (yr in years) {
    df_year <- df[df$year == yr, , drop = FALSE]
    p_kw <- NA_real_
    kw_data <- df_year[!is.na(df_year$DMFT_Index) & !is.na(df_year$abuse), , drop = FALSE]
    if (length(unique(kw_data$abuse)) >= 2) {
      kw <- try(kruskal.test(DMFT_Index ~ abuse, data = kw_data), silent = TRUE)
      if (!inherits(kw, "try-error")) p_kw <- kw$p.value
    }
    first_row <- TRUE
    for (abuse in abuse_types) {
      subset_df <- df_year[df_year$abuse == abuse, , drop = FALSE]
      if (nrow(subset_df) == 0) next
      row <- data.frame(Year = ifelse(first_row, as.character(yr), ""), Abuse_Type = abuse, N = nrow(subset_df), stringsAsFactors = FALSE)
      for (j in seq_along(vars_to_summarize)) {
        var_col <- vars_to_summarize[j]
        var_name <- names_to_summarize[j]
        x <- subset_df[[var_col]][!is.na(subset_df[[var_col]])]
        if (length(x) > 0) {
          row[[paste0(var_name, " Mean (SD)")]] <- sprintf("%.2f (%.2f)", mean(x), sd(x))
          row[[paste0(var_name, " Median [IQR]")]] <- sprintf("%.1f [%.1f-%.1f]", median(x), quantile(x, 0.25), quantile(x, 0.75))
        } else {
          row[[paste0(var_name, " Mean (SD)")]] <- "N/A"
          row[[paste0(var_name, " Median [IQR]")]] <- "N/A"
        }
      }
      row$`DMFT p-value (KW)` <- ifelse(first_row, ifelse(is.na(p_kw), "N/A", ifelse(p_kw < 0.0001, "<0.0001", sprintf("%.4f", p_kw))), "")
      table7_rows[[length(table7_rows) + 1]] <- row
      first_row <- FALSE
    }
  }
}
table7 <- if (length(table7_rows) > 0) bind_rows(table7_rows) else data.frame()
if (nrow(table7) > 0) write.csv(table7, file.path(OUTPUT_DIR, paste0("table7_dmft_by_year_abuse_", timestamp, ".csv")), row.names = FALSE)

# -----------------------------
# 15. Visualizations
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Creating visualizations...")

abuse_order <- c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")
df_plot <- df[df$abuse %in% abuse_order, , drop = FALSE]
if (nrow(df_plot) > 0 && "DMFT_Index" %in% names(df_plot)) {
  p <- ggplot(df_plot, aes(x = abuse, y = DMFT_Index)) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = 0.15, alpha = 0.35, size = 1) +
    scale_x_discrete(limits = abuse_order) +
    labs(x = "Abuse Type", y = "DMFT Index") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
  ggsave(file.path(OUTPUT_DIR, "figure1_dmft_boxplot.png"), p, width = 10, height = 6, dpi = 300)
}

for (var_name in c("gingivitis", "needTOBEtreated", "OralCleanStatus")) {
  if (!(var_name %in% names(df_plot))) next
  df_valid <- df_plot[!is.na(df_plot[[var_name]]) & !is.na(df_plot$abuse), , drop = FALSE]
  if (nrow(df_valid) == 0) next
  plot_counts <- as.data.frame(table(df_valid$abuse, df_valid[[var_name]]), stringsAsFactors = FALSE)
  names(plot_counts) <- c("abuse", "category", "n")
  plot_counts <- plot_counts %>% group_by(abuse) %>% mutate(percent = n / sum(n) * 100) %>% ungroup()
  p <- ggplot(plot_counts, aes(x = abuse, y = percent, fill = category)) +
    geom_col(position = "stack") +
    scale_x_discrete(limits = abuse_order) +
    labs(x = "Abuse Type", y = "Percentage (%)", fill = var_name) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
  ggsave(file.path(OUTPUT_DIR, paste0("figure_", var_name, "_bar.png")), p, width = 10, height = 6, dpi = 300)
}

# Overall dentition plot.
if ("dentition_type" %in% names(df) && "DMFT_Index" %in% names(df)) {
  df_dent_plot <- df[df$dentition_type %in% dentition_order & !is.na(df$DMFT_Index), , drop = FALSE]
  if (nrow(df_dent_plot) > 0) {
    p <- ggplot(df_dent_plot, aes(x = dentition_type, y = DMFT_Index)) +
      geom_boxplot(outlier.shape = NA) +
      geom_jitter(width = 0.15, alpha = 0.35, size = 1) +
      stat_summary(fun = mean, geom = "point", shape = 18, size = 3) +
      scale_x_discrete(limits = dentition_order) +
      labs(x = "Dentition Period", y = "Caries Experience") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 15, hjust = 1))
    ggsave(file.path(OUTPUT_DIR, paste0("figure_overall_dentition_", timestamp, ".png")), p, width = 10, height = 6, dpi = 300)
  }
}

# Abuse by dentition facet plot.
if ("dentition_type" %in% names(df_plot) && "DMFT_Index" %in% names(df_plot)) {
  df_facet <- df_plot[df_plot$dentition_type %in% dentition_order & !is.na(df_plot$DMFT_Index), , drop = FALSE]
  if (nrow(df_facet) > 0) {
    p <- ggplot(df_facet, aes(x = abuse, y = DMFT_Index)) +
      geom_boxplot(outlier.shape = NA) +
      geom_jitter(width = 0.15, alpha = 0.35, size = 0.8) +
      stat_summary(fun = mean, geom = "point", shape = 18, size = 2) +
      scale_x_discrete(limits = abuse_order) +
      facet_wrap(~ dentition_type, nrow = 1) +
      labs(x = "Abuse Type", y = "Caries Experience") +
      theme_minimal() +
      theme(axis.text.x = element_text(angle = 25, hjust = 1))
    ggsave(file.path(OUTPUT_DIR, paste0("figure_abuse_by_dentition_facet_", timestamp, ".png")), p, width = 14, height = 6, dpi = 300)
  }
}

# Pairwise boxplots for selected outcomes. Post-hoc tables are already written above.
for (var_name in c("Healthy_Rate", "Baby_d", "Baby_DMFT", "Care_Index", "UTN_Score", "DMFT_Index")) {
  if (!(var_name %in% names(df_plot))) next
  plot_data <- df_plot[!is.na(df_plot[[var_name]]) & !is.na(df_plot$abuse), , drop = FALSE]
  if (var_name %in% ratio_vars && "DMFT_Index" %in% names(plot_data)) plot_data <- plot_data[plot_data$DMFT_Index > 0, , drop = FALSE]
  if (nrow(plot_data) == 0) next
  p <- ggplot(plot_data, aes(x = abuse, y = .data[[var_name]])) +
    geom_boxplot(outlier.shape = NA) +
    geom_jitter(width = 0.15, alpha = 0.35, size = 1) +
    stat_summary(fun = mean, geom = "point", shape = 18, size = 3) +
    scale_x_discrete(limits = abuse_order) +
    labs(x = "Abuse Type", y = var_name) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 20, hjust = 1))
  if (var_name == "DMFT_Index") {
    plot_file <- file.path(OUTPUT_DIR, paste0("pairwise_results_DMFT_Index_", timestamp, ".png"))
  } else {
    plot_file <- file.path(OUTPUT_DIR, paste0("pairwise_results_", var_name, "_", timestamp, ".png"))
  }
  ggsave(plot_file, p, width = 10, height = 6, dpi = 300)
}

# -----------------------------
# 16. Summary report
# -----------------------------
sig_table <- data.frame()
if (nrow(t3_overall) > 0 && "Significant" %in% names(t3_overall)) {
  sig_table <- t3_overall[t3_overall$Significant == "Yes", , drop = FALSE]
}
summary_path <- file.path(OUTPUT_DIR, paste0("summary_report_", timestamp, ".txt"))
summary_lines <- c("Summary Report", paste0("Total N: ", nrow(df)), "Significant Differences:")
writeLines(summary_lines, summary_path)
if (nrow(sig_table) > 0) {
  capture.output(print(sig_table), file = summary_path, append = TRUE)
}
message("Summary saved to ", summary_path)

# -----------------------------
# 17. Sensitivity analysis: include multi-type records
# -----------------------------
if ("abuse_num" %in% names(df_all)) {
  message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Running sensitivity analysis including multi-type cases...")
  df_sens <- df_all
  df_sens$is_multitype <- as.integer(df_sens$abuse_num > 1)
  if (!is.null(subject_id_col) && subject_id_col %in% names(df_sens) && "date" %in% names(df_sens)) {
    df_sens <- df_sens[order(df_sens$date), , drop = FALSE]
    df_sens <- df_sens[!duplicated(df_sens[[subject_id_col]]), , drop = FALSE]
  }
  if ("abuse" %in% names(df_sens) && is.factor(df_sens$abuse)) df_sens$abuse <- droplevels(df_sens$abuse)

  # Recompute the same core derived oral-health variables directly for sensitivity dataset.
  if ("age_year" %in% names(df_sens)) {
    df_sens$age_group <- cut(df_sens$age_year, breaks = c(0, 6, 12, 18), labels = c("Early Childhood (2-6)", "Middle Childhood (7-12)", "Adolescence (13-18)"), right = TRUE, include.lowest = TRUE)
  }
  perm_cols_sens <- perm_teeth_cols[perm_teeth_cols %in% names(df_sens)]
  baby_cols_sens <- baby_teeth_cols[baby_teeth_cols %in% names(df_sens)]
  for (tc in c(perm_cols_sens, baby_cols_sens)) df_sens[[tc]] <- suppressWarnings(as.numeric(df_sens[[tc]]))
  if (length(perm_cols_sens) > 0) {
    pm <- df_sens[, perm_cols_sens, drop = FALSE]
    pm_all_na <- rowSums(!is.na(pm)) == 0
    df_sens$Perm_D <- rowSums(pm == 3, na.rm = TRUE); df_sens$Perm_D[pm_all_na] <- NA_real_
    df_sens$Perm_M <- rowSums(pm == 4, na.rm = TRUE); df_sens$Perm_M[pm_all_na] <- NA_real_
    df_sens$Perm_F <- rowSums(pm == 1, na.rm = TRUE); df_sens$Perm_F[pm_all_na] <- NA_real_
    df_sens$Perm_Sound <- rowSums(pm == 0, na.rm = TRUE); df_sens$Perm_Sound[pm_all_na] <- NA_real_
    df_sens$Perm_C0 <- rowSums(pm == 2, na.rm = TRUE); df_sens$Perm_C0[pm_all_na] <- NA_real_
    df_sens$Perm_DMFT <- df_sens$Perm_D + df_sens$Perm_M + df_sens$Perm_F
    df_sens$Perm_DMFT_C0 <- df_sens$Perm_DMFT + df_sens$Perm_C0
    df_sens$Perm_total_teeth <- rowSums(!is.na(pm) & pm != -1, na.rm = TRUE)
  } else {
    df_sens$Perm_D <- NA_real_; df_sens$Perm_M <- NA_real_; df_sens$Perm_F <- NA_real_; df_sens$Perm_Sound <- NA_real_; df_sens$Perm_C0 <- NA_real_; df_sens$Perm_DMFT <- NA_real_; df_sens$Perm_DMFT_C0 <- NA_real_; df_sens$Perm_total_teeth <- 0
  }
  if (length(baby_cols_sens) > 0) {
    bm <- df_sens[, baby_cols_sens, drop = FALSE]
    bm_all_na <- rowSums(!is.na(bm)) == 0
    df_sens$Baby_d <- rowSums(bm == 3, na.rm = TRUE); df_sens$Baby_d[bm_all_na] <- NA_real_
    df_sens$Baby_m <- rowSums(bm == 4, na.rm = TRUE); df_sens$Baby_m[bm_all_na] <- NA_real_
    df_sens$Baby_f <- rowSums(bm == 1, na.rm = TRUE); df_sens$Baby_f[bm_all_na] <- NA_real_
    df_sens$Baby_sound <- rowSums(bm == 0, na.rm = TRUE); df_sens$Baby_sound[bm_all_na] <- NA_real_
    df_sens$Baby_C0 <- rowSums(bm == 2, na.rm = TRUE); df_sens$Baby_C0[bm_all_na] <- NA_real_
    df_sens$Baby_DMFT <- df_sens$Baby_d + df_sens$Baby_m + df_sens$Baby_f
    df_sens$Baby_DMFT_C0 <- df_sens$Baby_DMFT + df_sens$Baby_C0
    df_sens$Baby_total_teeth <- rowSums(!is.na(bm) & bm != -1, na.rm = TRUE)
  } else {
    df_sens$Baby_d <- NA_real_; df_sens$Baby_m <- NA_real_; df_sens$Baby_f <- NA_real_; df_sens$Baby_sound <- NA_real_; df_sens$Baby_C0 <- NA_real_; df_sens$Baby_DMFT <- NA_real_; df_sens$Baby_DMFT_C0 <- NA_real_; df_sens$Baby_total_teeth <- 0
  }
  df_sens$DMFT_Index <- ifelse(is.na(df_sens$Perm_DMFT), 0, df_sens$Perm_DMFT) + ifelse(is.na(df_sens$Baby_DMFT), 0, df_sens$Baby_DMFT)
  df_sens$DMFT_Index[is.na(df_sens$Perm_DMFT) & is.na(df_sens$Baby_DMFT)] <- NA_real_
  df_sens$filled_total <- ifelse(is.na(df_sens$Perm_F), 0, df_sens$Perm_F) + ifelse(is.na(df_sens$Baby_f), 0, df_sens$Baby_f)
  df_sens$decayed_total <- ifelse(is.na(df_sens$Perm_D), 0, df_sens$Perm_D) + ifelse(is.na(df_sens$Baby_d), 0, df_sens$Baby_d)
  df_sens$total_teeth <- ifelse(is.na(df_sens$Perm_total_teeth), 0, df_sens$Perm_total_teeth) + ifelse(is.na(df_sens$Baby_total_teeth), 0, df_sens$Baby_total_teeth)
  df_sens$has_caries <- as.integer(!is.na(df_sens$DMFT_Index) & df_sens$DMFT_Index > 0)
  df_sens$has_untreated_caries <- as.integer(!is.na(df_sens$decayed_total) & df_sens$decayed_total > 0)
  if ("date" %in% names(df_sens)) df_sens$year <- as.integer(format(df_sens$date, "%Y"))
  if ("sex" %in% names(df_sens)) df_sens$sex_male <- as.integer(df_sens$sex == "Male")
  if ("gingivitis" %in% names(df_sens)) df_sens$gingivitis_binary <- as.integer(df_sens$gingivitis == "Gingivitis")
  if ("needTOBEtreated" %in% names(df_sens)) df_sens$treatment_need <- as.integer(df_sens$needTOBEtreated == "Treatment Required")

  sens_outcome_vars <- c("has_caries", "has_untreated_caries")
  sens_outcome_labels <- c("Caries Experience (>0)", "Untreated Caries")
  if ("gingivitis_binary" %in% names(df_sens)) { sens_outcome_vars <- c(sens_outcome_vars, "gingivitis_binary"); sens_outcome_labels <- c(sens_outcome_labels, "Gingivitis") }
  if ("treatment_need" %in% names(df_sens)) { sens_outcome_vars <- c(sens_outcome_vars, "treatment_need"); sens_outcome_labels <- c(sens_outcome_labels, "Treatment Need") }
  table4_sens_rows <- list()
  for (out_i in seq_along(sens_outcome_vars)) {
    outcome_var <- sens_outcome_vars[out_i]
    outcome_label <- sens_outcome_labels[out_i]
    for (comparison in comparison_categories) {
      df_model <- df_sens[df_sens$abuse %in% c(reference_category, comparison), , drop = FALSE]
      if (!("age_year" %in% names(df_model)) || !("sex_male" %in% names(df_model))) next
      df_model$comparison <- as.integer(df_model$abuse == comparison)
      needed_cols <- c(outcome_var, "age_year", "sex_male", "comparison", "abuse", "is_multitype")
      if ("year" %in% names(df_model)) needed_cols <- c(needed_cols, "year")
      if (!is.null(examiner_col) && examiner_col %in% names(df_model)) needed_cols <- c(needed_cols, examiner_col)
      needed_cols <- unique(needed_cols[needed_cols %in% names(df_model)])
      df_model <- df_model[, needed_cols, drop = FALSE]
      df_model <- df_model[complete.cases(df_model[, c(outcome_var, "age_year", "sex_male", "comparison", "is_multitype"), drop = FALSE]), , drop = FALSE]
      if (nrow(df_model) < 50) next
      if (length(unique(df_model[[outcome_var]])) < 2) next
      rhs_terms <- c("splines::ns(age_year, df = 4)", "sex_male", "comparison", "is_multitype")
      adjusted_for <- c("Age (spline)", "Sex", "is_multitype")
      if ("year" %in% names(df_model)) { rhs_terms <- c(rhs_terms, "factor(year)"); adjusted_for <- c(adjusted_for, "Year (FE)") }
      if (!is.null(examiner_col) && examiner_col %in% names(df_model)) { rhs_terms <- c(rhs_terms, paste0("factor(", examiner_col, ")")); adjusted_for <- c(adjusted_for, "Examiner (FE)") }
      model_formula <- as.formula(paste(outcome_var, "~", paste(rhs_terms, collapse = " + ")))
      fit <- try(glm(model_formula, data = df_model, family = binomial()), silent = TRUE)
      beta <- NA_real_; se <- NA_real_; p_val <- NA_real_; model_name <- "Logit (glm)"
      if (!inherits(fit, "try-error")) {
        coefs <- summary(fit)$coefficients
        if ("comparison" %in% rownames(coefs)) { beta <- coefs["comparison", "Estimate"]; se <- coefs["comparison", "Std. Error"]; p_val <- coefs["comparison", "Pr(>|z|)"] }
      }
      if ((is.na(beta) || is.na(se) || !is.finite(beta) || !is.finite(se)) && has_logistf) {
        fit_firth <- try(logistf::logistf(model_formula, data = df_model), silent = TRUE)
        if (!inherits(fit_firth, "try-error")) { beta <- fit_firth$coefficients["comparison"]; se <- sqrt(diag(fit_firth$var))["comparison"]; p_val <- fit_firth$prob["comparison"]; model_name <- "Logit (Firth/logistf)" }
      }
      or_val <- exp(beta); ci_low <- exp(beta - 1.96 * se); ci_up <- exp(beta + 1.96 * se)
      table4_sens_rows[[length(table4_sens_rows) + 1]] <- data.frame(
        Stratum = "",
        Outcome = outcome_label,
        Comparison = paste0(comparison, " vs ", reference_category),
        N = nrow(df_model),
        Events = sum(df_model[[outcome_var]], na.rm = TRUE),
        `Odds Ratio` = ifelse(is.finite(or_val), sprintf("%.2f", or_val), "N/A"),
        `95% CI` = ifelse(is.finite(ci_low) & is.finite(ci_up), sprintf("(%.2f-%.2f)", ci_low, ci_up), "N/A"),
        `p-value` = ifelse(is.na(p_val), "N/A", ifelse(p_val < 0.0001, "<0.0001", sprintf("%.4f", p_val))),
        Model = model_name,
        Adjusted_for = paste(adjusted_for, collapse = ", "),
        check.names = FALSE,
        stringsAsFactors = FALSE
      )
    }
  }
  table4_sens <- if (length(table4_sens_rows) > 0) bind_rows(table4_sens_rows) else data.frame()
  if (nrow(table4_sens) > 0) write.csv(table4_sens, file.path(OUTPUT_DIR, paste0("table4_logistic_regression_sensitivity_multitype_", timestamp, ".csv")), row.names = FALSE)
}

message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Analysis complete. Results saved to ", OUTPUT_DIR)
