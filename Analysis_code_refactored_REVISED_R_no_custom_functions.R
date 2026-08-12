# -----------------------------
# 0. Packages
# -----------------------------
# install.packages("logistf", repos = "https://cloud.r-project.org")
required_packages <- c("dplyr", "tidyr", "ggplot2", "splines", "logistf")
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
suppressPackageStartupMessages(library(logistf))

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

# BASE_DIR <- normalizePath(SCRIPT_DIR, mustWork = FALSE)
BASE_DIR <- "/Users/ayo/Desktop/_GSAIS_/Research/OralHealth_tokyo/paper_analysis"

# BASE_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = FALSE)
DATA_DIR <- file.path(BASE_DIR, "data")
DATA_DESCRIPTION_OUTPUT_DIR <- file.path(DATA_DIR, "data_description")
OUTPUT_DIR <- file.path(BASE_DIR, "result", timestamp)

dir.create(DATA_DESCRIPTION_OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

message("SCRIPT_DIR: ", SCRIPT_DIR)
message("BASE_DIR: ", BASE_DIR)
message("DATA_DIR: ", DATA_DIR)
message("OUTPUT_DIR: ", OUTPUT_DIR)

ORIGINAL_DATA_NAME <- "analysisData_20260211"
ORIGINAL_DATA_PATH <- file.path(DATA_DIR, paste0(ORIGINAL_DATA_NAME, ".csv"))
message("ORIGINAL_DATA_PATH: ", ORIGINAL_DATA_PATH)

END_DATE <- as.Date("2024-03-31")
target_abuse_types <- c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")
SUBJECT_ID_COL_CANDIDATES <- c("No_All", "child_id", "subject_id", "case_id", "ID", "id")

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

if ("abuse_1" %in% names(data0)) {
  data0$abuse_1 <- as.character(data0$abuse_1)
  data0$abuse_1[data0$abuse_1 == "1"] <- "Physical Abuse"
  data0$abuse_1[data0$abuse_1 == "2"] <- "Neglect"
  data0$abuse_1[data0$abuse_1 == "3"] <- "Emotional Abuse"
  data0$abuse_1[data0$abuse_1 == "4"] <- "Sexual Abuse"
  data0$abuse_1[data0$abuse_1 == "5"] <- "Delinquency"
  data0$abuse_1[data0$abuse_1 == "6"] <- "Parenting Difficulties"
  data0$abuse_1[data0$abuse_1 == "7"] <- "Others"
  data0$abuse_1 <- factor(data0$abuse_1, levels = c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse", "Delinquency", "Parenting Difficulties", "Others"), ordered = TRUE)
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
exclude_cols <- c("No_All", "instruction_detail", "instruction", "memo", "Orthodontics", "dentists", "dental_hygienist", "wake_up", "breakfast", "morning_brushing", "school", "bedtime", "night_brushing", "TV", "game", "meal", "extra_lesson")
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
# Loaded raw	2480
# Date <= 2024-03-31	2162
# Target maltreatment (abuse in 4 types) & abuse_num>=1	1305
# Single-type only (abuse_num==1)	1235
# Multi-type excluded (abuse_num>1)	70
# Deduplicated to first exam per No_All	1235
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

  # 永久歯として存在する歯の本数
  # -1 = 未萌出/存在しない
  df$Perm_total_teeth <- rowSums(
    !is.na(perm_mat) & perm_mat != -1,
    na.rm = TRUE
  )

  # 永久歯が1本も存在しない症例
  no_perm_teeth <- df$Perm_total_teeth == 0

  # 各状態の歯数
  df$Perm_D <- rowSums(perm_mat == 3, na.rm = TRUE)
  df$Perm_M <- rowSums(perm_mat == 4, na.rm = TRUE)
  df$Perm_F <- rowSums(perm_mat == 1, na.rm = TRUE)
  df$Perm_Sound <- rowSums(perm_mat == 0, na.rm = TRUE)
  df$Perm_C0 <- rowSums(perm_mat == 2, na.rm = TRUE)

  # 永久歯が0本なら、永久歯に関する指標はNA
  df$Perm_D[no_perm_teeth] <- NA_real_
  df$Perm_M[no_perm_teeth] <- NA_real_
  df$Perm_F[no_perm_teeth] <- NA_real_
  df$Perm_Sound[no_perm_teeth] <- NA_real_
  df$Perm_C0[no_perm_teeth] <- NA_real_

  # Permanent DMFT
  df$Perm_DMFT <- df$Perm_D + df$Perm_M + df$Perm_F
  df$Perm_DMFT_C0 <- df$Perm_DMFT + df$Perm_C0

  # Sound rate
  df$Perm_sound_rate <- df$Perm_Sound / df$Perm_total_teeth * 100
  df$Perm_sound_rate[
    is.infinite(df$Perm_sound_rate) |
    df$Perm_total_teeth <= 0
  ] <- NA_real_

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
if ("abuse_num" %in% names(data0)) {
  df_multi <- data0[data0$abuse_num != 1, , drop = FALSE]
  if (nrow(df_multi) > 0) {
    df_multi_prof <- df_multi
    write.csv(df_multi_prof, file.path(DATA_DIR, paste0(csv_name, "_multi_type_profile.csv")), row.names = FALSE)
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
      df_multi_prof$Perm_D <- rowSums(pm == 3, na.rm = TRUE)
      df_multi_prof$Perm_D[pm_all_na] <- NA_real_
      df_multi_prof$Perm_M <- rowSums(pm == 4, na.rm = TRUE)
      df_multi_prof$Perm_M[pm_all_na] <- NA_real_
      df_multi_prof$Perm_F <- rowSums(pm == 1, na.rm = TRUE)
      df_multi_prof$Perm_F[pm_all_na] <- NA_real_
      df_multi_prof$Perm_Sound <- rowSums(pm == 0, na.rm = TRUE)
      df_multi_prof$Perm_Sound[pm_all_na] <- NA_real_
      df_multi_prof$Perm_C0 <- rowSums(pm == 2, na.rm = TRUE)
      df_multi_prof$Perm_C0[pm_all_na] <- NA_real_
      df_multi_prof$Perm_DMFT <- df_multi_prof$Perm_D + df_multi_prof$Perm_M + df_multi_prof$Perm_F
      df_multi_prof$Perm_total_teeth <- rowSums(!is.na(pm) & pm != -1, na.rm = TRUE)
    } else {
      df_multi_prof$Perm_D <- NA_real_
      df_multi_prof$Perm_M <- NA_real_
      df_multi_prof$Perm_F <- NA_real_
      df_multi_prof$Perm_Sound <- NA_real_
      df_multi_prof$Perm_C0 <- NA_real_
      df_multi_prof$Perm_DMFT <- NA_real_
      df_multi_prof$Perm_total_teeth <- 0
    }
    if (length(baby_cols_multi) > 0) {
      bm <- df_multi_prof[, baby_cols_multi, drop = FALSE]
      bm_all_na <- rowSums(!is.na(bm)) == 0
      df_multi_prof$Baby_d <- rowSums(bm == 3, na.rm = TRUE)
      df_multi_prof$Baby_d[bm_all_na] <- NA_real_
      df_multi_prof$Baby_m <- rowSums(bm == 4, na.rm = TRUE)
      df_multi_prof$Baby_m[bm_all_na] <- NA_real_
      df_multi_prof$Baby_f <- rowSums(bm == 1, na.rm = TRUE)
      df_multi_prof$Baby_f[bm_all_na] <- NA_real_
      df_multi_prof$Baby_sound <- rowSums(bm == 0, na.rm = TRUE)
      df_multi_prof$Baby_sound[bm_all_na] <- NA_real_
      df_multi_prof$Baby_C0 <- rowSums(bm == 2, na.rm = TRUE)
      df_multi_prof$Baby_C0[bm_all_na] <- NA_real_
      df_multi_prof$Baby_DMFT <- df_multi_prof$Baby_d + df_multi_prof$Baby_m + df_multi_prof$Baby_f
      df_multi_prof$Baby_total_teeth <- rowSums(!is.na(bm) & bm != -1, na.rm = TRUE)
    } else {
      df_multi_prof$Baby_d <- NA_real_
      df_multi_prof$Baby_m <- NA_real_
      df_multi_prof$Baby_f <- NA_real_
      df_multi_prof$Baby_sound <- NA_real_
      df_multi_prof$Baby_C0 <- NA_real_
      df_multi_prof$Baby_DMFT <- NA_real_
      df_multi_prof$Baby_total_teeth <- 0
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
# ============================================================
# Table 4 and Figure 4
# Pairwise logistic regression models
#
# Reference category: Physical Abuse
# Comparisons:
#   1. Neglect vs Physical Abuse
#   2. Emotional Abuse vs Physical Abuse
#   3. Sexual Abuse vs Physical Abuse
#
# Adjustment variables:
#   - Age: restricted/natural cubic spline with df = 4
#   - Sex
#
# No adjustment for:
#   - Year
#   - Examiner
#   - Subject ID
# ============================================================

message(
  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  " - INFO - Creating Table 4 and Figure 4..."
)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)


# ------------------------------------------------------------
# 1. Create binary covariates and outcomes
# ------------------------------------------------------------

# Sex:
# Male = 1, other sex category = 0, missing = NA
if (!"sex_male" %in% names(df)) {
  if (!"sex" %in% names(df)) {
    stop("Variable `sex` was not found in df.")
  }

  df$sex_male <- ifelse(
    is.na(df$sex),
    NA_integer_,
    as.integer(df$sex == "Male")
  )
}


# Gingivitis:
# Gingivitis = 1, absent = 0, missing = NA
if (!"gingivitis_binary" %in% names(df)) {
  if (!"gingivitis" %in% names(df)) {
    stop("Variable `gingivitis` was not found in df.")
  }

  df$gingivitis_binary <- ifelse(
    is.na(df$gingivitis),
    NA_integer_,
    as.integer(df$gingivitis == "Gingivitis")
  )
}


# Treatment need:
# Treatment Required = 1, not required = 0, missing = NA
if (!"treatment_need" %in% names(df)) {
  if (!"needTOBEtreated" %in% names(df)) {
    stop("Variable `needTOBEtreated` was not found in df.")
  }

  df$treatment_need <- ifelse(
    is.na(df$needTOBEtreated),
    NA_integer_,
    as.integer(df$needTOBEtreated == "Treatment Required")
  )
}


# `has_caries` and `has_untreated_caries` must have been defined
# earlier in the script as:
#
# has_caries:
#   total caries experience > 0
#
# has_untreated_caries:
#   total number of untreated decayed teeth >= 1
#
# Example only:
# df$has_caries <- as.integer(total_caries_experience > 0)
# df$has_untreated_caries <- as.integer(total_decayed_teeth >= 1)


# ------------------------------------------------------------
# 2. Check required variables
# ------------------------------------------------------------

required_vars <- c(
  "abuse",
  "age_year",
  "sex_male",
  "has_caries",
  "has_untreated_caries",
  "gingivitis_binary",
  "treatment_need"
)

missing_vars <- setdiff(required_vars, names(df))

if (length(missing_vars) > 0) {
  stop(
    "The following required variables were not found in df: ",
    paste(missing_vars, collapse = ", ")
  )
}

if (!is.numeric(df$age_year)) {
  stop("`age_year` must be a numeric variable.")
}


# ------------------------------------------------------------
# 3. Ensure that all binary variables are coded as 0/1
# ------------------------------------------------------------

as_binary_01 <- function(x, variable_name) {

  if (is.logical(x)) {
    return(as.integer(x))
  }

  if (is.factor(x)) {
    x <- as.character(x)
  }

  original_nonmissing <- !is.na(x)
  x_numeric <- suppressWarnings(as.numeric(x))

  conversion_failed <- original_nonmissing & is.na(x_numeric)

  if (any(conversion_failed)) {
    stop(
      "`", variable_name,
      "` contains values that cannot be converted to 0/1."
    )
  }

  observed_values <- unique(x_numeric[!is.na(x_numeric)])

  if (!all(observed_values %in% c(0, 1))) {
    stop(
      "`", variable_name,
      "` must contain only 0, 1, and NA. Observed values: ",
      paste(observed_values, collapse = ", ")
    )
  }

  as.integer(x_numeric)
}

binary_vars <- c(
  "sex_male",
  "has_caries",
  "has_untreated_caries",
  "gingivitis_binary",
  "treatment_need"
)

for (variable_name in binary_vars) {
  df[[variable_name]] <- as_binary_01(
    df[[variable_name]],
    variable_name
  )
}


# ------------------------------------------------------------
# 4. Specify outcomes and pairwise comparisons
# ------------------------------------------------------------

outcome_vars <- c(
  "has_caries",
  "has_untreated_caries",
  "gingivitis_binary",
  "treatment_need"
)

outcome_labels <- c(
  has_caries = "Caries Experience (>0)",
  has_untreated_caries = "Untreated Caries",
  gingivitis_binary = "Gingivitis",
  treatment_need = "Treatment Need"
)

reference_category <- "Physical Abuse"

comparison_categories <- c(
  "Neglect",
  "Emotional Abuse",
  "Sexual Abuse"
)

expected_abuse_categories <- c(
  reference_category,
  comparison_categories
)

missing_abuse_categories <- setdiff(
  expected_abuse_categories,
  unique(df$abuse)
)

if (length(missing_abuse_categories) > 0) {
  warning(
    "The following maltreatment categories were not found in df: ",
    paste(missing_abuse_categories, collapse = ", ")
  )
}


# ------------------------------------------------------------
# 5. Fit pairwise logistic regression models
# ------------------------------------------------------------

table4_rows <- list()

for (outcome_var in outcome_vars) {

  outcome_label <- unname(outcome_labels[[outcome_var]])

  for (comparison_category in comparison_categories) {

    # Retain only Physical Abuse and the comparison category
    df_model <- df[
      df$abuse %in% c(reference_category, comparison_category),
      c(
        "abuse",
        outcome_var,
        "age_year",
        "sex_male"
      ),
      drop = FALSE
    ]

    # Complete-case analysis for variables included in this model
    df_model <- df_model[
      complete.cases(
        df_model[, c(
          "abuse",
          outcome_var,
          "age_year",
          "sex_male"
        )]
      ),
      ,
      drop = FALSE
    ]

    # Physical Abuse = 0
    # Comparison category = 1
    df_model$comparison <- as.integer(
      df_model$abuse == comparison_category
    )

    model_n <- nrow(df_model)

    # Group-specific denominators and events among the complete cases
    # actually included in this pairwise model.
    reference_n <- sum(
      df_model$abuse == reference_category,
      na.rm = TRUE
    )

    comparison_n <- sum(
      df_model$abuse == comparison_category,
      na.rm = TRUE
    )

    reference_events <- sum(
      df_model$abuse == reference_category &
        df_model[[outcome_var]] == 1L,
      na.rm = TRUE
    )

    comparison_events <- sum(
      df_model$abuse == comparison_category &
        df_model[[outcome_var]] == 1L,
      na.rm = TRUE
    )

    beta <- NA_real_
    standard_error <- NA_real_
    p_value <- NA_real_

    adjusted_or <- NA_real_
    ci_lower <- NA_real_
    ci_upper <- NA_real_

    model_status <- "Not fitted"

    # Verify that both maltreatment categories are represented
    if (model_n == 0) {

      model_status <- "No complete cases"

    } else if (length(unique(df_model$comparison)) < 2) {

      model_status <- "Only one maltreatment category present"

    } else if (length(unique(df_model[[outcome_var]])) < 2) {

      model_status <- "Outcome has no variation"

    } else {

      # splines::ns() fits a natural cubic spline, i.e.,
      # a restricted cubic spline with linear tails.
      #
      # Only age and sex are included as adjustment variables.
      model_formula <- stats::as.formula(
        paste0(
          outcome_var,
          " ~ comparison",
          " + splines::ns(age_year, df = 4)",
          " + sex_male"
        )
      )

      fit_result <- tryCatch(
        {
          list(
            fit = stats::glm(
              formula = model_formula,
              data = df_model,
              family = stats::binomial(link = "logit"),
              control = stats::glm.control(maxit = 100)
            ),
            error = NULL
          )
        },
        error = function(e) {
          list(
            fit = NULL,
            error = conditionMessage(e)
          )
        }
      )

      fit <- fit_result$fit

      if (is.null(fit)) {

        model_status <- paste0(
          "Model error: ",
          fit_result$error
        )

      } else if (!isTRUE(fit$converged)) {

        model_status <- "Model did not converge"

      } else {

        coefficient_table <- summary(fit)$coefficients

        if (!"comparison" %in% rownames(coefficient_table)) {

          model_status <- "Comparison coefficient unavailable"

        } else {

          beta <- coefficient_table[
            "comparison",
            "Estimate"
          ]

          standard_error <- coefficient_table[
            "comparison",
            "Std. Error"
          ]

          p_value <- coefficient_table[
            "comparison",
            "Pr(>|z|)"
          ]

          if (
            is.finite(beta) &&
            is.finite(standard_error)
          ) {

            critical_value <- stats::qnorm(0.975)

            adjusted_or <- exp(beta)

            ci_lower <- exp(
              beta - critical_value * standard_error
            )

            ci_upper <- exp(
              beta + critical_value * standard_error
            )
          }

          if (isTRUE(fit$boundary)) {
            model_status <- paste0(
              "Converged; boundary fit/",
              "possible separation"
            )
          } else {
            model_status <- "Converged"
          }
        }
      }
    }

    table4_rows[[length(table4_rows) + 1L]] <- data.frame(
      Outcome = outcome_label,
      Comparison = paste0(
        comparison_category,
        " vs ",
        reference_category
      ),
      Reference_group = reference_category,
      Comparison_group = comparison_category,
      Reference_N = reference_n,
      Reference_Events = reference_events,
      Comparison_N = comparison_n,
      Comparison_Events = comparison_events,
      Analysis_N = model_n,
      OR_numeric = adjusted_or,
      CI_lower_numeric = ci_lower,
      CI_upper_numeric = ci_upper,
      p_numeric = p_value,
      Model = "Pairwise logistic regression (glm)",
      Adjusted_for = paste0(
        "Age (restricted cubic spline, df = 4), ",
        "Sex"
      ),
      Model_status = model_status,
      stringsAsFactors = FALSE
    )
  }
}


# ------------------------------------------------------------
# 6. Combine and format Table 4
# ------------------------------------------------------------

if (length(table4_rows) == 0) {
  stop("No pairwise logistic regression results were produced.")
}

table4_numeric <- dplyr::bind_rows(table4_rows)


format_estimate <- function(x) {
  output <- rep("N/A", length(x))
  valid <- is.finite(x)
  output[valid] <- sprintf("%.2f", x[valid])
  output
}

format_confidence_interval <- function(lower, upper) {
  output <- rep("N/A", length(lower))

  valid <- (
    is.finite(lower) &
    is.finite(upper)
  )

  output[valid] <- sprintf(
    "(%.2f-%.2f)",
    lower[valid],
    upper[valid]
  )

  output
}

format_p_value <- function(x) {
  output <- rep("N/A", length(x))

  valid <- is.finite(x)

  output[valid & x < 0.0001] <- "<0.0001"

  regular <- valid & x >= 0.0001

  output[regular] <- sprintf(
    "%.4f",
    x[regular]
  )

  output
}


format_events_n_percent <- function(events, n) {
  output <- rep("N/A", length(n))

  valid <- (
    !is.na(events) &
    !is.na(n) &
    n > 0
  )

  output[valid] <- sprintf(
    "%d/%d (%.1f%%)",
    as.integer(events[valid]),
    as.integer(n[valid]),
    100 * events[valid] / n[valid]
  )

  output
}


table4_output <- data.frame(
  Outcome = table4_numeric$Outcome,
  `Reference group` = table4_numeric$Reference_group,
  `Reference events/N (%)` = format_events_n_percent(
    table4_numeric$Reference_Events,
    table4_numeric$Reference_N
  ),
  `Comparison group` = table4_numeric$Comparison_group,
  `Comparison events/N (%)` = format_events_n_percent(
    table4_numeric$Comparison_Events,
    table4_numeric$Comparison_N
  ),
  `Pairwise model N` = table4_numeric$Analysis_N,
  `Adjusted Odds Ratio` = format_estimate(
    table4_numeric$OR_numeric
  ),
  `95% CI` = format_confidence_interval(
    table4_numeric$CI_lower_numeric,
    table4_numeric$CI_upper_numeric
  ),
  `p-value` = format_p_value(
    table4_numeric$p_numeric
  ),
  Model = table4_numeric$Model,
  Adjusted_for = table4_numeric$Adjusted_for,
  Model_status = table4_numeric$Model_status,
  check.names = FALSE,
  stringsAsFactors = FALSE
)


table4_file <- file.path(
  OUTPUT_DIR,
  paste0(
    "table4_pairwise_logistic_regression_group_counts_",
    timestamp,
    ".csv"
  )
)

utils::write.csv(
  table4_output,
  table4_file,
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

message(
  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  " - INFO - Table 4 saved to: ",
  table4_file
)


# =============================================================================
# 7. Figure 4: common data preparation
# =============================================================================

# Use the unrounded numeric estimates stored in table4_numeric.
# This avoids reconstructing ORs and confidence limits from formatted strings.
if (nrow(table4_numeric) == 0) {
  stop("Figure 4 could not be generated because table4_numeric is empty.")
}

forest_base <- table4_numeric |>
  dplyr::filter(
    is.finite(OR_numeric),
    is.finite(CI_lower_numeric),
    is.finite(CI_upper_numeric),
    OR_numeric > 0,
    CI_lower_numeric > 0,
    CI_upper_numeric > 0
  ) |>
  dplyr::mutate(
    OR = OR_numeric,
    CI_lower = CI_lower_numeric,
    CI_upper = CI_upper_numeric,
    p_num = p_numeric
  )

if (nrow(forest_base) == 0) {
  stop("Figure 4 could not be generated because no finite estimates were available.")
}

comparison_short_map <- c(
  "Neglect vs Physical Abuse" = "Neglect",
  "Emotional Abuse vs Physical Abuse" = "Emotional Abuse",
  "Sexual Abuse vs Physical Abuse" = "Sexual Abuse"
)

# Standardize the outcome labels used in the manuscript figure.
outcome_display_map_en <- c(
  "Any caries experience (>0)" = "Caries Experience (>0)",
  "Caries Experience (>0)" = "Caries Experience (>0)",
  "Any untreated caries (>=1 decayed tooth)" = "Untreated Caries",
  "Untreated Caries" = "Untreated Caries",
  "Gingivitis" = "Gingivitis",
  "Treatment need" = "Treatment Need",
  "Treatment Need" = "Treatment Need"
)

# Use the actual group sizes in df rather than hard-coded sample sizes.
n_map <- stats::setNames(
  vapply(
    comparison_categories,
    function(g) {
      as.integer(sum(as.character(df$abuse) == g, na.rm = TRUE))
    },
    integer(1)
  ),
  comparison_categories
)

forest_base <- forest_base |>
  dplyr::mutate(
    Comp_short = unname(comparison_short_map[Comparison]),
    Outcome_display_en = unname(outcome_display_map_en[Outcome]),
    n_group = unname(n_map[Comp_short]),
    OR_CI_text = sprintf(
      "%.2f (%.2f-%.2f)",
      OR,
      CI_lower,
      CI_upper
    ),
    sig_shape = ifelse(
      !is.na(p_num) & p_num < 0.05,
      "sig",
      "ns"
    )
  )

if (anyNA(forest_base$Comp_short)) {
  stop(
    "Figure 4 contains an unrecognized comparison label: ",
    paste(
      unique(forest_base$Comparison[is.na(forest_base$Comp_short)]),
      collapse = ", "
    )
  )
}

if (anyNA(forest_base$Outcome_display_en)) {
  stop(
    "Figure 4 contains an unrecognized outcome label: ",
    paste(
      unique(forest_base$Outcome[is.na(forest_base$Outcome_display_en)]),
      collapse = ", "
    )
  )
}

if (anyNA(forest_base$n_group)) {
  stop("Figure 4 could not determine one or more maltreatment-group sample sizes.")
}

# Fixed manuscript ordering.
outcome_order_en <- c(
  "Caries Experience (>0)",
  "Untreated Caries",
  "Gingivitis",
  "Treatment Need"
)

comp_order_en <- c(
  "Neglect",
  "Emotional Abuse",
  "Sexual Abuse"
)

# y positions create visible gaps between the four outcome blocks.
y_positions <- c(
  15, 14, 13,
  11, 10, 9,
  7, 6, 5,
  3, 2, 1
)

shape_map <- c(
  "sig" = 15,
  "ns" = 16
)

# Wong colorblind-friendly palette.
comp_color_map_en <- c(
  "Neglect" = "#E69F00",
  "Emotional Abuse" = "#56B4E9",
  "Sexual Abuse" = "#009E73"
)

expected_figure_rows <- length(outcome_order_en) * length(comp_order_en)

if (nrow(forest_base) != expected_figure_rows) {
  stop(
    "Figure 4 requires exactly ",
    expected_figure_rows,
    " estimable outcome-comparison rows, but ",
    nrow(forest_base),
    " were available."
  )
}


# =============================================================================
# Figure 4A: English version
# Outcome heading is placed on the Neglect row (top row of each outcome block)
# =============================================================================

forest_en <- forest_base |>
  dplyr::mutate(
    Outcome_en = factor(
      Outcome_display_en,
      levels = outcome_order_en
    ),
    Comp_short = factor(
      Comp_short,
      levels = comp_order_en
    ),
    y_label = paste0(
      as.character(Comp_short),
      "\n(n=",
      n_group,
      ")"
    )
  ) |>
  dplyr::arrange(Outcome_en, Comp_short)

if (anyNA(forest_en$Outcome_en) || anyNA(forest_en$Comp_short)) {
  stop("Unexpected factor level detected in Figure 4 English data.")
}

if (anyDuplicated(forest_en[c("Outcome_en", "Comp_short")]) > 0) {
  stop("Duplicate outcome-comparison rows were detected in Figure 4 English data.")
}

forest_en$y <- y_positions

# Put each outcome heading at the same vertical position as Neglect.
group_label_positions_en <- forest_en |>
  dplyr::filter(Comp_short == "Neglect") |>
  dplyr::transmute(
    Outcome_en,
    y_label_position = y
  )

p_forest_en <- ggplot2::ggplot(
  forest_en,
  ggplot2::aes(x = OR, y = y)
) +
  ggplot2::geom_vline(
    xintercept = 1,
    linetype = "dashed",
    color = "black",
    linewidth = 0.7
  ) +
  ggplot2::geom_errorbarh(
    ggplot2::aes(
      xmin = CI_lower,
      xmax = CI_upper,
      color = Comp_short
    ),
    height = 0.10,
    linewidth = 1.0
  ) +
  ggplot2::geom_point(
    ggplot2::aes(
      color = Comp_short,
      shape = sig_shape
    ),
    size = 3.5
  ) +
  ggplot2::geom_text(
    ggplot2::aes(
      x = 2.95,
      label = OR_CI_text
    ),
    hjust = 0,
    size = 4.2,
    color = "black"
  ) +
  ggplot2::geom_text(
    data = group_label_positions_en,
    ggplot2::aes(
      x = -1.35,
      y = y_label_position,
      label = as.character(Outcome_en)
    ),
    inherit.aes = FALSE,
    hjust = 1,
    vjust = 0.5,
    fontface = "bold",
    size = 5.0
  ) +
  ggplot2::scale_color_manual(
    values = comp_color_map_en,
    guide = "none"
  ) +
  ggplot2::scale_shape_manual(
    values = shape_map,
    guide = "none"
  ) +
  ggplot2::scale_y_continuous(
    breaks = forest_en$y,
    labels = forest_en$y_label,
    expand = ggplot2::expansion(mult = c(0.02, 0.02))
  ) +
  ggplot2::scale_x_continuous(
    breaks = seq(0, 4.5, by = 0.5),
    expand = c(0, 0)
  ) +
  ggplot2::coord_cartesian(
    xlim = c(0, 4.5),
    clip = "off"
  ) +
  ggplot2::labs(
    x = "Odds Ratio (95% CI)",
    y = NULL
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    panel.grid.major.y = ggplot2::element_blank(),
    panel.grid.minor = ggplot2::element_blank(),
    panel.grid.major.x = ggplot2::element_blank(),
    panel.background = ggplot2::element_rect(
      fill = "transparent",
      color = NA
    ),
    plot.background = ggplot2::element_rect(
      fill = "transparent",
      color = NA
    ),
    panel.border = ggplot2::element_rect(
      color = "black",
      fill = NA,
      linewidth = 0.8
    ),
    axis.text.y = ggplot2::element_text(
      size = 10.5,
      color = "black"
    ),
    axis.text.x = ggplot2::element_text(
      size = 11,
      color = "black"
    ),
    axis.title.x = ggplot2::element_text(
      size = 15,
      face = "bold",
      color = "black"
    ),
    plot.margin = ggplot2::margin(
      t = 20,
      r = 180,
      b = 20,
      l = 280
    )
  )

# Main English manuscript figure.
figure4_main_file <- file.path(
  OUTPUT_DIR,
  paste0("figure_forest_plot_", timestamp, ".png")
)

ggplot2::ggsave(
  filename = figure4_main_file,
  plot = p_forest_en,
  width = 11,
  height = 10,
  dpi = 300,
  bg = "transparent"
)

# Explicitly named English copy.
figure4_english_file <- file.path(
  OUTPUT_DIR,
  paste0(
    "figure_forest_plot_english_style_transparent_",
    timestamp,
    ".png"
  )
)

ggplot2::ggsave(
  filename = figure4_english_file,
  plot = p_forest_en,
  width = 11,
  height = 10,
  dpi = 300,
  bg = "transparent"
)

message(
  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  " - INFO - English Figure 4 saved to: ",
  figure4_main_file
)


# =============================================================================
# Figure 4B: Japanese version
# =============================================================================

outcome_label_map_ja <- c(
  "Caries Experience (>0)" = "う蝕経験あり（>0）",
  "Untreated Caries" = "未処置う蝕あり",
  "Gingivitis" = "歯肉炎",
  "Treatment Need" = "歯科治療の必要性"
)

comparison_label_map_ja <- c(
  "Neglect" = "ネグレクト",
  "Emotional Abuse" = "心理的虐待",
  "Sexual Abuse" = "性的虐待"
)

forest_ja <- forest_en |>
  dplyr::mutate(
    Outcome_ja = factor(
      unname(outcome_label_map_ja[as.character(Outcome_en)]),
      levels = unname(outcome_label_map_ja[outcome_order_en])
    ),
    Comp_ja = factor(
      unname(comparison_label_map_ja[as.character(Comp_short)]),
      levels = unname(comparison_label_map_ja[comp_order_en])
    ),
    y_label_ja = paste0(
      as.character(Comp_ja),
      "\n(n=",
      n_group,
      ")"
    )
  )

group_label_positions_ja <- forest_ja |>
  dplyr::filter(Comp_short == "Neglect") |>
  dplyr::transmute(
    Outcome_ja,
    y_label_position = y
  )

p_forest_ja <- ggplot2::ggplot(
  forest_ja,
  ggplot2::aes(x = OR, y = y)
) +
  ggplot2::geom_vline(
    xintercept = 1,
    linetype = "dashed",
    color = "black",
    linewidth = 0.7
  ) +
  ggplot2::geom_errorbarh(
    ggplot2::aes(
      xmin = CI_lower,
      xmax = CI_upper,
      color = Comp_short
    ),
    height = 0.10,
    linewidth = 1.0
  ) +
  ggplot2::geom_point(
    ggplot2::aes(
      color = Comp_short,
      shape = sig_shape
    ),
    size = 3.5
  ) +
  ggplot2::geom_text(
    ggplot2::aes(
      x = 2.95,
      label = OR_CI_text
    ),
    hjust = 0,
    size = 4.2,
    color = "black"
  ) +
  ggplot2::geom_text(
    data = group_label_positions_ja,
    ggplot2::aes(
      x = -1.35,
      y = y_label_position,
      label = as.character(Outcome_ja)
    ),
    inherit.aes = FALSE,
    hjust = 1,
    vjust = 0.5,
    fontface = "bold",
    size = 5.0
  ) +
  ggplot2::scale_color_manual(
    values = comp_color_map_en,
    guide = "none"
  ) +
  ggplot2::scale_shape_manual(
    values = shape_map,
    guide = "none"
  ) +
  ggplot2::scale_y_continuous(
    breaks = forest_ja$y,
    labels = forest_ja$y_label_ja,
    expand = ggplot2::expansion(mult = c(0.02, 0.02))
  ) +
  ggplot2::scale_x_continuous(
    breaks = seq(0, 4.5, by = 0.5),
    expand = c(0, 0)
  ) +
  ggplot2::coord_cartesian(
    xlim = c(0, 4.5),
    clip = "off"
  ) +
  ggplot2::labs(
    x = "調整オッズ比（95%信頼区間）",
    y = NULL
  ) +
  ggplot2::theme_minimal() +
  ggplot2::theme(
    panel.grid.major.y = ggplot2::element_blank(),
    panel.grid.minor = ggplot2::element_blank(),
    panel.grid.major.x = ggplot2::element_blank(),
    panel.background = ggplot2::element_rect(
      fill = "transparent",
      color = NA
    ),
    plot.background = ggplot2::element_rect(
      fill = "transparent",
      color = NA
    ),
    panel.border = ggplot2::element_rect(
      color = "black",
      fill = NA,
      linewidth = 0.8
    ),
    axis.text.y = ggplot2::element_text(
      size = 10.5,
      color = "black"
    ),
    axis.text.x = ggplot2::element_text(
      size = 11,
      color = "black"
    ),
    axis.title.x = ggplot2::element_text(
      size = 15,
      face = "bold",
      color = "black"
    ),
    plot.margin = ggplot2::margin(
      t = 20,
      r = 180,
      b = 20,
      l = 280
    )
  )

figure4_japanese_file <- file.path(
  OUTPUT_DIR,
  paste0(
    "figure_forest_plot_japanese_style_transparent_",
    timestamp,
    ".png"
  )
)

ggplot2::ggsave(
  filename = figure4_japanese_file,
  plot = p_forest_ja,
  width = 11,
  height = 10,
  dpi = 300,
  bg = "transparent"
)

message(
  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  " - INFO - Japanese Figure 4 saved to: ",
  figure4_japanese_file
)

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
          abuse1 <- abuse_types[i_abuse]
          abuse2 <- abuse_types[j_abuse]
          p_adj <- NA_real_
          p_unadj <- NA_real_
          if (abuse1 %in% rownames(p_adj_matrix) && abuse2 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[abuse1, abuse2]
          if (abuse2 %in% rownames(p_adj_matrix) && abuse1 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[abuse2, abuse1]
          if (abuse1 %in% rownames(p_unadj_matrix) && abuse2 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[abuse1, abuse2]
          if (abuse2 %in% rownames(p_unadj_matrix) && abuse1 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[abuse2, abuse1]
          if (is.na(p_adj)) next
          vals1 <- df_dent$DMFT_Index[df_dent$abuse == abuse1]
          vals2 <- df_dent$DMFT_Index[df_dent$abuse == abuse2]
          vals1 <- vals1[!is.na(vals1)]
          vals2 <- vals2[!is.na(vals2)]
          if (length(vals1) == 0 || length(vals2) == 0) next
          q1 <- quantile(vals1, c(0.25, 0.75))
          q2 <- quantile(vals2, c(0.25, 0.75))
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
  p_adj_matrix <- NULL
  p_unadj_matrix <- NULL
  posthoc_label <- "Dunn (PMCMRplus)"
  if (has_PMCMRplus) {
    dunn_adj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_abuse$DMFT_Index, g = df_abuse$dentition_type, p.adjust.method = "bonferroni"), silent = TRUE)
    dunn_unadj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_abuse$DMFT_Index, g = df_abuse$dentition_type, p.adjust.method = "none"), silent = TRUE)
    if (!inherits(dunn_adj, "try-error") && !inherits(dunn_unadj, "try-error")) {
      p_adj_matrix <- dunn_adj$p.value
      p_unadj_matrix <- dunn_unadj$p.value
    }
  }
  if (is.null(p_adj_matrix)) {
    posthoc_label <- "Pairwise Wilcoxon fallback"
    pw_adj <- try(pairwise.wilcox.test(df_abuse$DMFT_Index, df_abuse$dentition_type, p.adjust.method = "bonferroni", exact = FALSE), silent = TRUE)
    pw_unadj <- try(pairwise.wilcox.test(df_abuse$DMFT_Index, df_abuse$dentition_type, p.adjust.method = "none", exact = FALSE), silent = TRUE)
    if (!inherits(pw_adj, "try-error") && !inherits(pw_unadj, "try-error")) {
      p_adj_matrix <- pw_adj$p.value
      p_unadj_matrix <- pw_unadj$p.value
    }
  }
  if (is.null(p_adj_matrix)) next
  for (i_dent in seq_len(length(dentition_order) - 1)) {
    for (j_dent in seq((i_dent + 1), length(dentition_order))) {
      dent1 <- dentition_order[i_dent]
      dent2 <- dentition_order[j_dent]
      p_adj <- NA_real_
      p_unadj <- NA_real_
      if (dent1 %in% rownames(p_adj_matrix) && dent2 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[dent1, dent2]
      if (dent2 %in% rownames(p_adj_matrix) && dent1 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[dent2, dent1]
      if (dent1 %in% rownames(p_unadj_matrix) && dent2 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[dent1, dent2]
      if (dent2 %in% rownames(p_unadj_matrix) && dent1 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[dent2, dent1]
      if (is.na(p_adj)) next
      vals1 <- df_abuse$DMFT_Index[df_abuse$dentition_type == dent1]
      vals2 <- df_abuse$DMFT_Index[df_abuse$dentition_type == dent2]
      vals1 <- vals1[!is.na(vals1)]
      vals2 <- vals2[!is.na(vals2)]
      if (length(vals1) == 0 || length(vals2) == 0) next
      q1 <- quantile(vals1, c(0.25, 0.75))
      q2 <- quantile(vals2, c(0.25, 0.75))
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
      p_adj_matrix <- NULL
      p_unadj_matrix <- NULL
      posthoc_label <- "Dunn (PMCMRplus)"
      if (has_PMCMRplus) {
        dunn_adj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_overall_dent$DMFT_Index, g = df_overall_dent$dentition_type, p.adjust.method = "bonferroni"), silent = TRUE)
        dunn_unadj <- try(PMCMRplus::kwAllPairsDunnTest(x = df_overall_dent$DMFT_Index, g = df_overall_dent$dentition_type, p.adjust.method = "none"), silent = TRUE)
        if (!inherits(dunn_adj, "try-error") && !inherits(dunn_unadj, "try-error")) {
          p_adj_matrix <- dunn_adj$p.value
          p_unadj_matrix <- dunn_unadj$p.value
        }
      }
      if (is.null(p_adj_matrix)) {
        posthoc_label <- "Pairwise Wilcoxon fallback"
        pw_adj <- try(pairwise.wilcox.test(df_overall_dent$DMFT_Index, df_overall_dent$dentition_type, p.adjust.method = "bonferroni", exact = FALSE), silent = TRUE)
        pw_unadj <- try(pairwise.wilcox.test(df_overall_dent$DMFT_Index, df_overall_dent$dentition_type, p.adjust.method = "none", exact = FALSE), silent = TRUE)
        if (!inherits(pw_adj, "try-error") && !inherits(pw_unadj, "try-error")) {
          p_adj_matrix <- pw_adj$p.value
          p_unadj_matrix <- pw_unadj$p.value
        }
      }
      if (!is.null(p_adj_matrix)) {
        for (i_dent in seq_len(length(dentition_order) - 1)) {
          for (j_dent in seq((i_dent + 1), length(dentition_order))) {
            dent1 <- dentition_order[i_dent]
            dent2 <- dentition_order[j_dent]
            p_adj <- NA_real_
            p_unadj <- NA_real_
            if (dent1 %in% rownames(p_adj_matrix) && dent2 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[dent1, dent2]
            if (dent2 %in% rownames(p_adj_matrix) && dent1 %in% colnames(p_adj_matrix)) p_adj <- p_adj_matrix[dent2, dent1]
            if (dent1 %in% rownames(p_unadj_matrix) && dent2 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[dent1, dent2]
            if (dent2 %in% rownames(p_unadj_matrix) && dent1 %in% colnames(p_unadj_matrix)) p_unadj <- p_unadj_matrix[dent2, dent1]
            if (is.na(p_adj)) next
            vals1 <- df_overall_dent$DMFT_Index[df_overall_dent$dentition_type == dent1]
            vals2 <- df_overall_dent$DMFT_Index[df_overall_dent$dentition_type == dent2]
            vals1 <- vals1[!is.na(vals1)]
            vals2 <- vals2[!is.na(vals2)]
            if (length(vals1) == 0 || length(vals2) == 0) next
            q1 <- quantile(vals1, c(0.25, 0.75))
            q2 <- quantile(vals2, c(0.25, 0.75))
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
# 14. Supplementary Table S7: primary dmft and permanent DMFT separately
#     compared across maltreatment categories
# -----------------------------
message(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - INFO - Creating Supplementary Table S7...")

# Primary dmft is evaluated among children with at least one primary tooth,
# including children with primary or mixed dentition. Permanent DMFT is
# evaluated among children with at least one permanent tooth, including
# children with mixed or permanent dentition. This prevents children with no
# teeth of the relevant dentition from being treated as having a score of zero.
tableS7_specs <- data.frame(
  Outcome = c(
    "Primary dentition caries experience (dmft)",
    "Permanent dentition caries experience (DMFT)"
  ),
  Variable = c("Baby_DMFT", "Perm_DMFT"),
  Teeth_Count_Variable = c("Baby_total_teeth", "Perm_total_teeth"),
  Analysis_Population = c(
    "Children with >=1 primary tooth (primary or mixed dentition)",
    "Children with >=1 permanent tooth (mixed or permanent dentition)"
  ),
  stringsAsFactors = FALSE
)

tableS7_abuse_types <- target_abuse_types[
  target_abuse_types %in% unique(as.character(df$abuse[!is.na(df$abuse)]))
]

tableS7_rows <- list()
tableS7_posthoc_rows <- list()

for (s7_i in seq_len(nrow(tableS7_specs))) {
  s7_outcome <- tableS7_specs$Outcome[s7_i]
  s7_var <- tableS7_specs$Variable[s7_i]
  s7_teeth_var <- tableS7_specs$Teeth_Count_Variable[s7_i]
  s7_population <- tableS7_specs$Analysis_Population[s7_i]

  if (!(s7_var %in% names(df)) || !(s7_teeth_var %in% names(df)) || !("abuse" %in% names(df))) {
    warning("Table S7 skipped for ", s7_outcome, ": required variable(s) missing.")
    next
  }

  df_s7 <- df[
    !is.na(df[[s7_var]]) &
      !is.na(df[[s7_teeth_var]]) &
      df[[s7_teeth_var]] > 0 &
      !is.na(df$abuse) &
      df$abuse %in% tableS7_abuse_types,
    ,
    drop = FALSE
  ]

  if (nrow(df_s7) == 0) {
    warning("Table S7 skipped for ", s7_outcome, ": no eligible observations.")
    next
  }

  df_s7$abuse <- factor(as.character(df_s7$abuse), levels = tableS7_abuse_types)
  df_s7$abuse <- droplevels(df_s7$abuse)
  s7_present_groups <- levels(df_s7$abuse)

  if (length(s7_present_groups) < 2) {
    warning("Table S7 skipped for ", s7_outcome, ": fewer than two maltreatment categories.")
    next
  }

  kw_s7 <- try(
    kruskal.test(as.formula(paste(s7_var, "~ abuse")), data = df_s7),
    silent = TRUE
  )
  s7_kw_h <- NA_real_
  s7_kw_df <- NA_real_
  s7_kw_p <- NA_real_
  if (!inherits(kw_s7, "try-error")) {
    s7_kw_h <- as.numeric(kw_s7$statistic)
    s7_kw_df <- as.numeric(kw_s7$parameter)
    s7_kw_p <- kw_s7$p.value
  }

  s7_total <- df_s7[[s7_var]]
  s7_total <- s7_total[!is.na(s7_total)]
  s7_total_sd <- if (length(s7_total) > 1) sd(s7_total) else NA_real_
  s7_total_q <- if (length(s7_total) > 0) quantile(s7_total, c(0.25, 0.75), na.rm = TRUE) else c(NA_real_, NA_real_)

  s7_row <- data.frame(
    Outcome = s7_outcome,
    Analysis_population = s7_population,
    Total_N = length(s7_total),
    Total_Mean_SD = ifelse(
      length(s7_total) > 0,
      ifelse(
        is.na(s7_total_sd),
        sprintf("%.2f ± N/A", mean(s7_total)),
        sprintf("%.2f ± %.2f", mean(s7_total), s7_total_sd)
      ),
      "N/A"
    ),
    Total_Median_IQR = ifelse(
      length(s7_total) > 0,
      sprintf("%.2f [%.2f-%.2f]", median(s7_total), s7_total_q[1], s7_total_q[2]),
      "N/A"
    ),
    check.names = FALSE,
    stringsAsFactors = FALSE
  )

  for (s7_abuse in tableS7_abuse_types) {
    s7_x <- df_s7[[s7_var]][df_s7$abuse == s7_abuse]
    s7_x <- s7_x[!is.na(s7_x)]
    s7_x_sd <- if (length(s7_x) > 1) sd(s7_x) else NA_real_
    s7_x_q <- if (length(s7_x) > 0) quantile(s7_x, c(0.25, 0.75), na.rm = TRUE) else c(NA_real_, NA_real_)

    s7_row[[paste0(s7_abuse, " n")]] <- length(s7_x)
    s7_row[[paste0(s7_abuse, " Mean ± SD")]] <- ifelse(
      length(s7_x) > 0,
      ifelse(
        is.na(s7_x_sd),
        sprintf("%.2f ± N/A", mean(s7_x)),
        sprintf("%.2f ± %.2f", mean(s7_x), s7_x_sd)
      ),
      "N/A"
    )
    s7_row[[paste0(s7_abuse, " Median [IQR]")]] <- ifelse(
      length(s7_x) > 0,
      sprintf("%.2f [%.2f-%.2f]", median(s7_x), s7_x_q[1], s7_x_q[2]),
      "N/A"
    )
  }

  s7_row[["Kruskal-Wallis H"]] <- ifelse(is.na(s7_kw_h), "N/A", sprintf("%.3f", s7_kw_h))
  s7_row[["Kruskal-Wallis df"]] <- ifelse(is.na(s7_kw_df), "N/A", sprintf("%.0f", s7_kw_df))
  s7_row[["Overall p-value"]] <- ifelse(
    is.na(s7_kw_p),
    "N/A",
    ifelse(s7_kw_p < 0.0001, "<0.0001", sprintf("%.4f", s7_kw_p))
  )

  # Create fixed columns for all six pairwise comparisons so that the two
  # outcome rows have the same publication-ready structure.
  if (length(tableS7_abuse_types) >= 2) {
    for (s7_g1_i in seq_len(length(tableS7_abuse_types) - 1)) {
      for (s7_g2_i in seq((s7_g1_i + 1), length(tableS7_abuse_types))) {
        s7_g1 <- tableS7_abuse_types[s7_g1_i]
        s7_g2 <- tableS7_abuse_types[s7_g2_i]
        s7_row[[paste0(s7_g1, " vs ", s7_g2, " adjusted p")]] <- "N/A"
      }
    }
  }

  if (is.na(s7_kw_p)) {
    s7_posthoc_method <- "Not performed because the overall Kruskal-Wallis test was unavailable"
  } else if (s7_kw_p < 0.05) {
    s7_posthoc_method <- "Post-hoc test unavailable"
  } else {
    s7_posthoc_method <- "Not performed because overall Kruskal-Wallis p >= 0.05"
  }
  s7_significant_pairs <- character(0)

  # Perform pairwise post-hoc tests only when the omnibus test is significant.
  # Dunn's test with Bonferroni adjustment is preferred; pairwise Wilcoxon with
  # Bonferroni adjustment is used when PMCMRplus is unavailable.
  if (!is.na(s7_kw_p) && s7_kw_p < 0.05) {
    df_s7$rank_value_s7 <- rank(df_s7[[s7_var]], ties.method = "average")
    s7_mean_rank_table <- aggregate(rank_value_s7 ~ abuse, data = df_s7, FUN = mean)

    s7_p_adj_matrix <- NULL
    s7_p_unadj_matrix <- NULL

    if (has_PMCMRplus) {
      s7_dunn_adj <- try(
        PMCMRplus::kwAllPairsDunnTest(
          x = df_s7[[s7_var]],
          g = df_s7$abuse,
          p.adjust.method = "bonferroni"
        ),
        silent = TRUE
      )
      s7_dunn_unadj <- try(
        PMCMRplus::kwAllPairsDunnTest(
          x = df_s7[[s7_var]],
          g = df_s7$abuse,
          p.adjust.method = "none"
        ),
        silent = TRUE
      )
      if (!inherits(s7_dunn_adj, "try-error") && !inherits(s7_dunn_unadj, "try-error")) {
        s7_p_adj_matrix <- s7_dunn_adj$p.value
        s7_p_unadj_matrix <- s7_dunn_unadj$p.value
        s7_posthoc_method <- "Dunn test with Bonferroni adjustment"
      }
    }

    if (is.null(s7_p_adj_matrix)) {
      s7_pw_adj <- try(
        pairwise.wilcox.test(
          x = df_s7[[s7_var]],
          g = df_s7$abuse,
          p.adjust.method = "bonferroni",
          exact = FALSE
        ),
        silent = TRUE
      )
      s7_pw_unadj <- try(
        pairwise.wilcox.test(
          x = df_s7[[s7_var]],
          g = df_s7$abuse,
          p.adjust.method = "none",
          exact = FALSE
        ),
        silent = TRUE
      )
      if (!inherits(s7_pw_adj, "try-error") && !inherits(s7_pw_unadj, "try-error")) {
        s7_p_adj_matrix <- s7_pw_adj$p.value
        s7_p_unadj_matrix <- s7_pw_unadj$p.value
        s7_posthoc_method <- "Pairwise Wilcoxon test with Bonferroni adjustment"
      }
    }

    if (!is.null(s7_p_adj_matrix)) {
      for (s7_g1_i in seq_len(length(s7_present_groups) - 1)) {
        for (s7_g2_i in seq((s7_g1_i + 1), length(s7_present_groups))) {
          s7_g1 <- s7_present_groups[s7_g1_i]
          s7_g2 <- s7_present_groups[s7_g2_i]

          s7_p_adj <- NA_real_
          s7_p_unadj <- NA_real_

          if (s7_g1 %in% rownames(s7_p_adj_matrix) && s7_g2 %in% colnames(s7_p_adj_matrix)) {
            s7_p_adj <- s7_p_adj_matrix[s7_g1, s7_g2]
          }
          if (s7_g2 %in% rownames(s7_p_adj_matrix) && s7_g1 %in% colnames(s7_p_adj_matrix)) {
            s7_p_adj <- s7_p_adj_matrix[s7_g2, s7_g1]
          }
          if (s7_g1 %in% rownames(s7_p_unadj_matrix) && s7_g2 %in% colnames(s7_p_unadj_matrix)) {
            s7_p_unadj <- s7_p_unadj_matrix[s7_g1, s7_g2]
          }
          if (s7_g2 %in% rownames(s7_p_unadj_matrix) && s7_g1 %in% colnames(s7_p_unadj_matrix)) {
            s7_p_unadj <- s7_p_unadj_matrix[s7_g2, s7_g1]
          }

          if (is.na(s7_p_adj)) next

          s7_p_adj_text <- ifelse(s7_p_adj < 0.0001, "<0.0001", sprintf("%.4f", s7_p_adj))
          s7_p_unadj_text <- ifelse(
            is.na(s7_p_unadj),
            "N/A",
            ifelse(s7_p_unadj < 0.0001, "<0.0001", sprintf("%.4f", s7_p_unadj))
          )
          s7_row[[paste0(s7_g1, " vs ", s7_g2, " adjusted p")]] <- s7_p_adj_text

          s7_vals1 <- df_s7[[s7_var]][df_s7$abuse == s7_g1]
          s7_vals2 <- df_s7[[s7_var]][df_s7$abuse == s7_g2]
          s7_vals1 <- s7_vals1[!is.na(s7_vals1)]
          s7_vals2 <- s7_vals2[!is.na(s7_vals2)]
          if (length(s7_vals1) == 0 || length(s7_vals2) == 0) next

          s7_q1 <- quantile(s7_vals1, c(0.25, 0.75), na.rm = TRUE)
          s7_q2 <- quantile(s7_vals2, c(0.25, 0.75), na.rm = TRUE)
          s7_mr1 <- s7_mean_rank_table$rank_value_s7[s7_mean_rank_table$abuse == s7_g1]
          s7_mr2 <- s7_mean_rank_table$rank_value_s7[s7_mean_rank_table$abuse == s7_g2]

          s7_direction <- "Equal mean ranks"
          if (length(s7_mr1) > 0 && length(s7_mr2) > 0) {
            if (s7_mr1 > s7_mr2) s7_direction <- paste0(s7_g1, " > ", s7_g2)
            if (s7_mr2 > s7_mr1) s7_direction <- paste0(s7_g2, " > ", s7_g1)
          }

          if (!is.na(s7_p_adj) && s7_p_adj < 0.05) {
            s7_significant_pairs <- c(
              s7_significant_pairs,
              paste0(
                s7_direction,
                " (adjusted p ",
                ifelse(substr(s7_p_adj_text, 1, 1) == "<", "", "= "),
                s7_p_adj_text,
                ")"
              )
            )
          }

          tableS7_posthoc_rows[[length(tableS7_posthoc_rows) + 1]] <- data.frame(
            Outcome = s7_outcome,
            Analysis_population = s7_population,
            Group1 = s7_g1,
            Group2 = s7_g2,
            Comparison = paste0(s7_g1, " vs ", s7_g2),
            Group1_n = length(s7_vals1),
            Group2_n = length(s7_vals2),
            Group1_Mean_SD = ifelse(
              length(s7_vals1) > 1,
              sprintf("%.2f ± %.2f", mean(s7_vals1), sd(s7_vals1)),
              sprintf("%.2f ± N/A", mean(s7_vals1))
            ),
            Group2_Mean_SD = ifelse(
              length(s7_vals2) > 1,
              sprintf("%.2f ± %.2f", mean(s7_vals2), sd(s7_vals2)),
              sprintf("%.2f ± N/A", mean(s7_vals2))
            ),
            Group1_Median_IQR = sprintf("%.2f [%.2f-%.2f]", median(s7_vals1), s7_q1[1], s7_q1[2]),
            Group2_Median_IQR = sprintf("%.2f [%.2f-%.2f]", median(s7_vals2), s7_q2[1], s7_q2[2]),
            Group1_Mean_Rank = ifelse(length(s7_mr1) > 0, round(s7_mr1, 2), NA_real_),
            Group2_Mean_Rank = ifelse(length(s7_mr2) > 0, round(s7_mr2, 2), NA_real_),
            Direction_based_on_mean_rank = s7_direction,
            Kruskal_Wallis_p = s7_kw_p,
            p_unadjusted = s7_p_unadj,
            p_adjusted_Bonferroni = s7_p_adj,
            `p-value (unadjusted)` = s7_p_unadj_text,
            `p-value (adjusted)` = s7_p_adj_text,
            Significant_after_adjustment = ifelse(s7_p_adj < 0.05, "Yes", "No"),
            Method = s7_posthoc_method,
            check.names = FALSE,
            stringsAsFactors = FALSE
          )
        }
      }
    }
  }

  s7_row[["Post-hoc method"]] <- s7_posthoc_method
  s7_row[["Significant pairwise comparisons"]] <- ifelse(
    length(s7_significant_pairs) > 0,
    paste(s7_significant_pairs, collapse = "; "),
    "None"
  )

  tableS7_rows[[length(tableS7_rows) + 1]] <- s7_row
}

tableS7 <- if (length(tableS7_rows) > 0) bind_rows(tableS7_rows) else data.frame()
tableS7_posthoc <- if (length(tableS7_posthoc_rows) > 0) bind_rows(tableS7_posthoc_rows) else data.frame()

if (nrow(tableS7) > 0) {
  write.csv(
    tableS7,
    file.path(OUTPUT_DIR, paste0("tableS7_primary_and_permanent_caries_by_abuse_", timestamp, ".csv")),
    row.names = FALSE,
    na = ""
  )
}

# Detailed pairwise results are saved separately as an audit/support file; the
# publication-ready Table S7 is the wide table written above.
if (nrow(tableS7_posthoc) > 0) {
  write.csv(
    tableS7_posthoc,
    file.path(OUTPUT_DIR, paste0("tableS7_pairwise_detail_", timestamp, ".csv")),
    row.names = FALSE,
    na = ""
  )
}

# -----------------------------
# 15. Table 7: DMFT, Dt, Mt, Ft by year and abuse type
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
# 16. Visualizations
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
  plot_counts <- plot_counts %>%
    group_by(abuse) %>%
    mutate(percent = n / sum(n) * 100) %>%
    ungroup()
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
      facet_wrap(~dentition_type, nrow = 1) +
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
# 17. Summary report
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
# 18. Sensitivity analysis: include multi-type records
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
    df_sens$Perm_D <- rowSums(pm == 3, na.rm = TRUE)
    df_sens$Perm_D[pm_all_na] <- NA_real_
    df_sens$Perm_M <- rowSums(pm == 4, na.rm = TRUE)
    df_sens$Perm_M[pm_all_na] <- NA_real_
    df_sens$Perm_F <- rowSums(pm == 1, na.rm = TRUE)
    df_sens$Perm_F[pm_all_na] <- NA_real_
    df_sens$Perm_Sound <- rowSums(pm == 0, na.rm = TRUE)
    df_sens$Perm_Sound[pm_all_na] <- NA_real_
    df_sens$Perm_C0 <- rowSums(pm == 2, na.rm = TRUE)
    df_sens$Perm_C0[pm_all_na] <- NA_real_
    df_sens$Perm_DMFT <- df_sens$Perm_D + df_sens$Perm_M + df_sens$Perm_F
    df_sens$Perm_DMFT_C0 <- df_sens$Perm_DMFT + df_sens$Perm_C0
    df_sens$Perm_total_teeth <- rowSums(!is.na(pm) & pm != -1, na.rm = TRUE)
  } else {
    df_sens$Perm_D <- NA_real_
    df_sens$Perm_M <- NA_real_
    df_sens$Perm_F <- NA_real_
    df_sens$Perm_Sound <- NA_real_
    df_sens$Perm_C0 <- NA_real_
    df_sens$Perm_DMFT <- NA_real_
    df_sens$Perm_DMFT_C0 <- NA_real_
    df_sens$Perm_total_teeth <- 0
  }
  if (length(baby_cols_sens) > 0) {
    bm <- df_sens[, baby_cols_sens, drop = FALSE]
    bm_all_na <- rowSums(!is.na(bm)) == 0
    df_sens$Baby_d <- rowSums(bm == 3, na.rm = TRUE)
    df_sens$Baby_d[bm_all_na] <- NA_real_
    df_sens$Baby_m <- rowSums(bm == 4, na.rm = TRUE)
    df_sens$Baby_m[bm_all_na] <- NA_real_
    df_sens$Baby_f <- rowSums(bm == 1, na.rm = TRUE)
    df_sens$Baby_f[bm_all_na] <- NA_real_
    df_sens$Baby_sound <- rowSums(bm == 0, na.rm = TRUE)
    df_sens$Baby_sound[bm_all_na] <- NA_real_
    df_sens$Baby_C0 <- rowSums(bm == 2, na.rm = TRUE)
    df_sens$Baby_C0[bm_all_na] <- NA_real_
    df_sens$Baby_DMFT <- df_sens$Baby_d + df_sens$Baby_m + df_sens$Baby_f
    df_sens$Baby_DMFT_C0 <- df_sens$Baby_DMFT + df_sens$Baby_C0
    df_sens$Baby_total_teeth <- rowSums(!is.na(bm) & bm != -1, na.rm = TRUE)
  } else {
    df_sens$Baby_d <- NA_real_
    df_sens$Baby_m <- NA_real_
    df_sens$Baby_f <- NA_real_
    df_sens$Baby_sound <- NA_real_
    df_sens$Baby_C0 <- NA_real_
    df_sens$Baby_DMFT <- NA_real_
    df_sens$Baby_DMFT_C0 <- NA_real_
    df_sens$Baby_total_teeth <- 0
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
  if ("gingivitis_binary" %in% names(df_sens)) {
    sens_outcome_vars <- c(sens_outcome_vars, "gingivitis_binary")
    sens_outcome_labels <- c(sens_outcome_labels, "Gingivitis")
  }
  if ("treatment_need" %in% names(df_sens)) {
    sens_outcome_vars <- c(sens_outcome_vars, "treatment_need")
    sens_outcome_labels <- c(sens_outcome_labels, "Treatment Need")
  }
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
      needed_cols <- unique(needed_cols[needed_cols %in% names(df_model)])
      df_model <- df_model[, needed_cols, drop = FALSE]
      df_model <- df_model[complete.cases(df_model[, c(outcome_var, "age_year", "sex_male", "comparison", "is_multitype"), drop = FALSE]), , drop = FALSE]
      if (nrow(df_model) < 50) next
      if (length(unique(df_model[[outcome_var]])) < 2) next
      rhs_terms <- c("splines::ns(age_year, df = 4)", "sex_male", "comparison", "is_multitype")
      adjusted_for <- c("Age (spline)", "Sex", "is_multitype")
      if ("year" %in% names(df_model)) {
        rhs_terms <- c(rhs_terms, "factor(year)")
        adjusted_for <- c(adjusted_for, "Year (FE)")
      }
      model_formula <- as.formula(paste(outcome_var, "~", paste(rhs_terms, collapse = " + ")))
      fit <- try(glm(model_formula, data = df_model, family = binomial()), silent = TRUE)
      beta <- NA_real_
      se <- NA_real_
      p_val <- NA_real_
      model_name <- "Logit (glm)"
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


# =============================================================================
# Supplementary Table S6
# Comparison of children with abuse_num == 1 versus abuse_num == 2
#
# Included group: abuse_num == 1
# Excluded group: abuse_num == 2 exactly
#
# Recommended placement:
# Replace the existing block beginning with
#   "Compact profile of excluded multi-type cases"
# after the main feature-engineering section.
#
# This code assumes that the following objects have already been created:
#   df_all, OUTPUT_DIR, timestamp, SUBJECT_ID_COL_CANDIDATES,
#   perm_teeth_cols, baby_teeth_cols
# =============================================================================

message(
  format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
  " - INFO - Creating Supplementary Table S6: abuse_num 1 vs 2..."
)

# -----------------------------------------------------------------------------
# 1. Select the two comparison groups from the same eligible source population
# -----------------------------------------------------------------------------

if (!exists("df_all")) {
  stop("Object `df_all` was not found. Run the filtering section first.")
}

if (!("abuse_num" %in% names(df_all))) {
  stop("Variable `abuse_num` was not found in df_all.")
}

if (!exists("perm_teeth_cols") || !exists("baby_teeth_cols")) {
  stop("Objects `perm_teeth_cols` and/or `baby_teeth_cols` were not found.")
}

if (!exists("SUBJECT_ID_COL_CANDIDATES")) {
  SUBJECT_ID_COL_CANDIDATES <- c(
    "No_All", "child_id", "subject_id", "case_id", "ID", "id"
  )
}

if (!exists("timestamp")) {
  timestamp <- format(Sys.Date(), "%Y%m%d")
}

if (!exists("OUTPUT_DIR")) {
  OUTPUT_DIR <- getwd()
}

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

df_s6_source <- df_all

df_s6_source$abuse_num_numeric <- suppressWarnings(
  as.numeric(as.character(df_s6_source$abuse_num))
)

df_s6 <- df_s6_source[
  !is.na(df_s6_source$abuse_num_numeric) &
    df_s6_source$abuse_num_numeric %in% c(1, 2),
  ,
  drop = FALSE
]

if (nrow(df_s6) == 0) {
  stop("No records with abuse_num == 1 or abuse_num == 2 were found.")
}

# -----------------------------------------------------------------------------
# 2. Keep the first examination for each child, using the same principle as the
#    primary analysis. Records with a missing subject ID are retained.
# -----------------------------------------------------------------------------

subject_id_col_s6 <- NULL
for (candidate in SUBJECT_ID_COL_CANDIDATES) {
  if (candidate %in% names(df_s6) && is.null(subject_id_col_s6)) {
    subject_id_col_s6 <- candidate
  }
}

df_s6$.s6_original_order <- seq_len(nrow(df_s6))

if (!is.null(subject_id_col_s6)) {
  if ("date" %in% names(df_s6)) {
    df_s6 <- df_s6[
      order(df_s6$date, df_s6$.s6_original_order, na.last = TRUE),
      ,
      drop = FALSE
    ]
  }

  s6_id <- trimws(as.character(df_s6[[subject_id_col_s6]]))
  s6_valid_id <- !is.na(s6_id) & nzchar(s6_id)
  s6_duplicate_id <- s6_valid_id & duplicated(s6_id)
  df_s6 <- df_s6[!s6_duplicate_id, , drop = FALSE]
} else {
  warning(
    "No subject-ID column was found. Table S6 was generated without deduplication."
  )
}

df_s6$.s6_original_order <- NULL

# -----------------------------------------------------------------------------
# 3. Derive oral-health variables identically for both groups
# -----------------------------------------------------------------------------

derive_s6_oral_health_variables <- function(dat) {

  sum_two_components <- function(x, y) {
    result <- ifelse(is.na(x), 0, x) + ifelse(is.na(y), 0, y)
    result[is.na(x) & is.na(y)] <- NA_real_
    result
  }

  recode_binary <- function(x, positive_values, negative_values, variable_name) {
    x_chr <- trimws(as.character(x))
    result <- rep(NA_integer_, length(x_chr))

    result[!is.na(x_chr) & x_chr %in% positive_values] <- 1L
    result[!is.na(x_chr) & x_chr %in% negative_values] <- 0L

    unknown <- !is.na(x_chr) & nzchar(x_chr) & is.na(result)
    if (any(unknown)) {
      warning(
        "Unrecognized non-missing values in `", variable_name, "`: ",
        paste(sort(unique(x_chr[unknown])), collapse = ", "),
        ". These values were treated as missing."
      )
    }

    result
  }

  perm_cols_s6 <- perm_teeth_cols[perm_teeth_cols %in% names(dat)]
  baby_cols_s6 <- baby_teeth_cols[baby_teeth_cols %in% names(dat)]

  if (length(perm_cols_s6) == 0) {
    stop("No permanent-tooth columns were found in the Table S6 dataset.")
  }

  if (length(baby_cols_s6) == 0) {
    stop("No primary-tooth columns were found in the Table S6 dataset.")
  }

  for (tooth_col in c(perm_cols_s6, baby_cols_s6)) {
    dat[[tooth_col]] <- suppressWarnings(as.numeric(dat[[tooth_col]]))
  }

  # Permanent teeth -----------------------------------------------------------
  perm_mat_s6 <- dat[, perm_cols_s6, drop = FALSE]
  perm_all_missing <- rowSums(!is.na(perm_mat_s6)) == 0

  dat$Perm_D <- rowSums(perm_mat_s6 == 3, na.rm = TRUE)
  dat$Perm_M <- rowSums(perm_mat_s6 == 4, na.rm = TRUE)
  dat$Perm_F <- rowSums(perm_mat_s6 == 1, na.rm = TRUE)
  dat$Perm_Sound <- rowSums(perm_mat_s6 == 0, na.rm = TRUE)
  dat$Perm_total_teeth <- rowSums(
    !is.na(perm_mat_s6) & perm_mat_s6 != -1,
    na.rm = TRUE
  )

  for (v in c("Perm_D", "Perm_M", "Perm_F", "Perm_Sound")) {
    dat[[v]][perm_all_missing] <- NA_real_
  }
  dat$Perm_total_teeth[perm_all_missing] <- NA_real_

  dat$Perm_DMFT <- dat$Perm_D + dat$Perm_M + dat$Perm_F

  # Permanent DMFT is defined only for children with at least one permanent
  # tooth. A child with no permanent teeth is not treated as DMFT = 0.
  dat$Perm_DMFT_analysis <- dat$Perm_DMFT
  dat$Perm_DMFT_analysis[
    is.na(dat$Perm_total_teeth) | dat$Perm_total_teeth <= 0
  ] <- NA_real_

  # Primary teeth -------------------------------------------------------------
  baby_mat_s6 <- dat[, baby_cols_s6, drop = FALSE]
  baby_all_missing <- rowSums(!is.na(baby_mat_s6)) == 0

  dat$Baby_d <- rowSums(baby_mat_s6 == 3, na.rm = TRUE)
  dat$Baby_m <- rowSums(baby_mat_s6 == 4, na.rm = TRUE)
  dat$Baby_f <- rowSums(baby_mat_s6 == 1, na.rm = TRUE)
  dat$Baby_sound <- rowSums(baby_mat_s6 == 0, na.rm = TRUE)
  dat$Baby_total_teeth <- rowSums(
    !is.na(baby_mat_s6) & baby_mat_s6 != -1,
    na.rm = TRUE
  )

  for (v in c("Baby_d", "Baby_m", "Baby_f", "Baby_sound")) {
    dat[[v]][baby_all_missing] <- NA_real_
  }
  dat$Baby_total_teeth[baby_all_missing] <- NA_real_

  dat$Baby_DMFT <- dat$Baby_d + dat$Baby_m + dat$Baby_f

  # Primary dmft is defined only for children with at least one primary tooth.
  dat$Baby_DMFT_analysis <- dat$Baby_DMFT
  dat$Baby_DMFT_analysis[
    is.na(dat$Baby_total_teeth) | dat$Baby_total_teeth <= 0
  ] <- NA_real_

  # Total caries and tooth-status counts --------------------------------------
  dat$DMFT_Index <- sum_two_components(
    dat$Perm_DMFT_analysis,
    dat$Baby_DMFT_analysis
  )

  dat$decayed_total <- sum_two_components(dat$Perm_D, dat$Baby_d)
  dat$missing_total <- sum_two_components(dat$Perm_M, dat$Baby_m)
  dat$filled_total <- sum_two_components(dat$Perm_F, dat$Baby_f)
  dat$sound_total <- sum_two_components(dat$Perm_Sound, dat$Baby_sound)
  dat$total_teeth <- sum_two_components(
    dat$Perm_total_teeth,
    dat$Baby_total_teeth
  )

  # Healthy teeth rate --------------------------------------------------------
  dat$Healthy_Rate <- dat$sound_total / dat$total_teeth * 100
  dat$Healthy_Rate[
    !is.finite(dat$Healthy_Rate) |
      is.na(dat$total_teeth) |
      dat$total_teeth <= 0
  ] <- NA_real_

  # Care index and untreated-caries rate are defined only among children with
  # caries experience (DMFT_Index > 0).
  dat$Care_Index <- dat$filled_total / dat$DMFT_Index * 100
  dat$Care_Index[
    !is.finite(dat$Care_Index) |
      is.na(dat$DMFT_Index) |
      dat$DMFT_Index <= 0
  ] <- NA_real_

  dat$UTN_Score <- dat$decayed_total / dat$DMFT_Index * 100
  dat$UTN_Score[
    !is.finite(dat$UTN_Score) |
      is.na(dat$DMFT_Index) |
      dat$DMFT_Index <= 0
  ] <- NA_real_

  # Binary variables: preserve missingness rather than recoding missing values
  # as absence of the outcome.
  dat$has_caries_s6 <- ifelse(
    is.na(dat$DMFT_Index),
    NA_integer_,
    as.integer(dat$DMFT_Index > 0)
  )

  dat$has_untreated_caries_s6 <- ifelse(
    is.na(dat$decayed_total),
    NA_integer_,
    as.integer(dat$decayed_total > 0)
  )

  if (!("sex" %in% names(dat))) {
    dat$female_binary_s6 <- NA_integer_
    warning("Variable `sex` was not found; Female sex will be missing.")
  } else {
    dat$female_binary_s6 <- recode_binary(
      dat$sex,
      positive_values = c("Female", "female", "F", "女性", "女"),
      negative_values = c("Male", "male", "M", "男性", "男"),
      variable_name = "sex"
    )
  }

  if (!("gingivitis" %in% names(dat))) {
    dat$gingivitis_binary_s6 <- NA_integer_
    warning("Variable `gingivitis` was not found.")
  } else {
    dat$gingivitis_binary_s6 <- recode_binary(
      dat$gingivitis,
      positive_values = c("Gingivitis", "2"),
      negative_values = c("No Gingivitis", "1"),
      variable_name = "gingivitis"
    )
  }

  if (!("needTOBEtreated" %in% names(dat))) {
    dat$treatment_required_s6 <- NA_integer_
    warning("Variable `needTOBEtreated` was not found.")
  } else {
    dat$treatment_required_s6 <- recode_binary(
      dat$needTOBEtreated,
      positive_values = c("Treatment Required", "2"),
      negative_values = c("No Treatment Required", "1"),
      variable_name = "needTOBEtreated"
    )
  }

  if (!("OralCleanStatus" %in% names(dat))) {
    dat$fair_or_poor_hygiene_s6 <- NA_integer_
    warning("Variable `OralCleanStatus` was not found.")
  } else {
    dat$fair_or_poor_hygiene_s6 <- recode_binary(
      dat$OralCleanStatus,
      positive_values = c("Poor", "Fair", "1", "2"),
      negative_values = c("Good", "3"),
      variable_name = "OralCleanStatus"
    )
  }

  dat
}

df_s6 <- derive_s6_oral_health_variables(df_s6)

df_s6$comparison_group <- factor(
  df_s6$abuse_num_numeric,
  levels = c(1, 2),
  labels = c(
    "Included (abuse_num = 1)",
    "Excluded (abuse_num = 2)"
  )
)

if (sum(df_s6$abuse_num_numeric == 1, na.rm = TRUE) == 0) {
  stop("The included group (abuse_num == 1) has no observations.")
}

if (sum(df_s6$abuse_num_numeric == 2, na.rm = TRUE) == 0) {
  stop("The excluded group (abuse_num == 2) has no observations.")
}

# Save the analysis dataset for audit/reproducibility.
write.csv(
  df_s6,
  file.path(
    OUTPUT_DIR,
    paste0("tableS6_analysis_dataset_abuse_num1_vs_2_", timestamp, ".csv")
  ),
  row.names = FALSE,
  na = "",
  fileEncoding = "UTF-8"
)

# -----------------------------------------------------------------------------
# 4. Variable specifications, in the requested order
# -----------------------------------------------------------------------------

s6_specs <- data.frame(
  Variable = c(
    "Age, years",
    "Female sex",
    "Total caries experience",
    "Permanent DMFT",
    "Primary dmft",
    "Decayed teeth (D+d)",
    "Missing teeth (M+m)",
    "Filled teeth (F+f)",
    "Healthy teeth rate (%)",
    "Care index (% among children with caries)",
    "Untreated caries rate (% among children with caries)",
    "Caries experience (dmft/DMFT > 0)",
    "Untreated caries present",
    "Gingivitis present",
    "Treatment required",
    "Fair or poor oral hygiene"
  ),
  Column = c(
    "age_year",
    "female_binary_s6",
    "DMFT_Index",
    "Perm_DMFT_analysis",
    "Baby_DMFT_analysis",
    "decayed_total",
    "missing_total",
    "filled_total",
    "Healthy_Rate",
    "Care_Index",
    "UTN_Score",
    "has_caries_s6",
    "has_untreated_caries_s6",
    "gingivitis_binary_s6",
    "treatment_required_s6",
    "fair_or_poor_hygiene_s6"
  ),
  Type = c(
    "continuous",
    "binary",
    "continuous",
    "continuous",
    "continuous",
    "continuous",
    "continuous",
    "continuous",
    "continuous",
    "continuous",
    "continuous",
    "binary",
    "binary",
    "binary",
    "binary",
    "binary"
  ),
  stringsAsFactors = FALSE
)

missing_s6_columns <- setdiff(s6_specs$Column, names(df_s6))
if (length(missing_s6_columns) > 0) {
  stop(
    "The following Table S6 variables were not created: ",
    paste(missing_s6_columns, collapse = ", ")
  )
}

# -----------------------------------------------------------------------------
# 5. Formatting and statistical-test helpers
# -----------------------------------------------------------------------------

format_s6_p <- function(p) {
  if (length(p) == 0 || is.na(p) || !is.finite(p)) {
    return("N/A")
  }
  if (p < 0.0001) {
    return("<0.0001")
  }
  sprintf("%.4f", p)
}

summarize_s6_continuous <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[!is.na(x) & is.finite(x)]
  n <- length(x)

  if (n == 0) {
    return(list(
      n = 0L,
      mean_sd = "N/A",
      median_iqr = "N/A",
      display = "N/A"
    ))
  }

  x_sd <- if (n > 1) stats::sd(x) else NA_real_
  x_q <- stats::quantile(x, probs = c(0.25, 0.75), na.rm = TRUE)

  mean_sd <- if (is.na(x_sd)) {
    sprintf("%.2f ± N/A", mean(x))
  } else {
    sprintf("%.2f ± %.2f", mean(x), x_sd)
  }

  median_iqr <- sprintf(
    "%.2f [%.2f-%.2f]",
    stats::median(x),
    x_q[1],
    x_q[2]
  )

  list(
    n = as.integer(n),
    mean_sd = mean_sd,
    median_iqr = median_iqr,
    display = paste0(mean_sd, "; ", median_iqr)
  )
}

summarize_s6_binary <- function(x) {
  x <- suppressWarnings(as.numeric(x))
  x <- x[!is.na(x)]
  n <- length(x)

  if (n == 0) {
    return(list(
      n = 0L,
      events = NA_integer_,
      percent = NA_real_,
      display = "N/A"
    ))
  }

  if (!all(x %in% c(0, 1))) {
    stop("A binary Table S6 variable contains values other than 0, 1, or NA.")
  }

  events <- sum(x == 1)
  percent <- 100 * events / n

  list(
    n = as.integer(n),
    events = as.integer(events),
    percent = percent,
    display = sprintf("%d/%d (%.1f%%)", events, n, percent)
  )
}

run_s6_continuous_test <- function(x, group) {
  valid <- !is.na(x) & is.finite(suppressWarnings(as.numeric(x))) &
    !is.na(group) & group %in% c(1, 2)

  x_valid <- suppressWarnings(as.numeric(x[valid]))
  group_valid <- group[valid]
  x_included <- x_valid[group_valid == 1]
  x_excluded <- x_valid[group_valid == 2]

  if (length(x_included) == 0 || length(x_excluded) == 0) {
    return(list(
      test = "Wilcoxon rank-sum test (Mann-Whitney U; two-sided)",
      statistic = "N/A",
      p = NA_real_,
      note = "Test unavailable because one group had no non-missing observations"
    ))
  }

  test_result <- try(
    suppressWarnings(
      stats::wilcox.test(
        x_included,
        x_excluded,
        alternative = "two.sided",
        exact = FALSE,
        correct = TRUE
      )
    ),
    silent = TRUE
  )

  if (inherits(test_result, "try-error")) {
    return(list(
      test = "Wilcoxon rank-sum test (Mann-Whitney U; two-sided)",
      statistic = "N/A",
      p = NA_real_,
      note = "Test failed"
    ))
  }

  list(
    test = "Wilcoxon rank-sum test (Mann-Whitney U; two-sided)",
    statistic = sprintf("W = %.1f", as.numeric(test_result$statistic)),
    p = as.numeric(test_result$p.value),
    note = ""
  )
}

run_s6_binary_test <- function(x, group) {
  valid <- !is.na(x) & !is.na(group) & group %in% c(1, 2)
  x_valid <- suppressWarnings(as.numeric(x[valid]))
  group_valid <- group[valid]

  if (!all(x_valid %in% c(0, 1))) {
    return(list(
      test = "N/A",
      statistic = "N/A",
      p = NA_real_,
      note = "Binary variable contained values other than 0 or 1"
    ))
  }

  n_included <- sum(group_valid == 1)
  n_excluded <- sum(group_valid == 2)

  if (n_included == 0 || n_excluded == 0) {
    return(list(
      test = "N/A",
      statistic = "N/A",
      p = NA_real_,
      note = "Test unavailable because one group had no non-missing observations"
    ))
  }

  if (length(unique(x_valid)) < 2) {
    return(list(
      test = "N/A",
      statistic = "N/A",
      p = NA_real_,
      note = "Test unavailable because the outcome had no variation"
    ))
  }

  contingency_table <- table(
    factor(group_valid, levels = c(1, 2)),
    factor(x_valid, levels = c(0, 1))
  )

  chi_check <- try(
    suppressWarnings(stats::chisq.test(contingency_table, correct = FALSE)),
    silent = TRUE
  )

  use_fisher <- inherits(chi_check, "try-error")
  if (!use_fisher) {
    use_fisher <- any(chi_check$expected < 5)
  }

  if (use_fisher) {
    fisher_result <- try(stats::fisher.test(contingency_table), silent = TRUE)

    if (inherits(fisher_result, "try-error")) {
      return(list(
        test = "Fisher's exact test (two-sided)",
        statistic = "N/A",
        p = NA_real_,
        note = "Test failed"
      ))
    }

    return(list(
      test = "Fisher's exact test (two-sided)",
      statistic = "Exact",
      p = as.numeric(fisher_result$p.value),
      note = "Used because at least one expected cell count was <5"
    ))
  }

  list(
    test = "Pearson chi-square test",
    statistic = sprintf(
      "Chi-square = %.3f",
      as.numeric(chi_check$statistic)
    ),
    p = as.numeric(chi_check$p.value),
    note = ""
  )
}

# -----------------------------------------------------------------------------
# 6. Generate Table S6
# -----------------------------------------------------------------------------

s6_rows <- list()

for (i in seq_len(nrow(s6_specs))) {
  variable_label <- s6_specs$Variable[i]
  variable_column <- s6_specs$Column[i]
  variable_type <- s6_specs$Type[i]

  included_values <- df_s6[[variable_column]][
    df_s6$abuse_num_numeric == 1
  ]
  excluded_values <- df_s6[[variable_column]][
    df_s6$abuse_num_numeric == 2
  ]

  if (variable_type == "continuous") {
    included_summary <- summarize_s6_continuous(included_values)
    excluded_summary <- summarize_s6_continuous(excluded_values)
    test_result <- run_s6_continuous_test(
      df_s6[[variable_column]],
      df_s6$abuse_num_numeric
    )

    summary_measure <- "Mean ± SD; Median [IQR]"

  } else if (variable_type == "binary") {
    included_summary <- summarize_s6_binary(included_values)
    excluded_summary <- summarize_s6_binary(excluded_values)
    test_result <- run_s6_binary_test(
      df_s6[[variable_column]],
      df_s6$abuse_num_numeric
    )

    summary_measure <- "Positive n/N (%)"

  } else {
    stop("Unknown Table S6 variable type: ", variable_type)
  }

  s6_rows[[length(s6_rows) + 1L]] <- data.frame(
    Variable = variable_label,
    `Summary measure` = summary_measure,
    `Included non-missing N` = included_summary$n,
    `Included: abuse_num = 1` = included_summary$display,
    `Excluded non-missing N` = excluded_summary$n,
    `Excluded: abuse_num = 2` = excluded_summary$display,
    Test = test_result$test,
    `Test statistic` = test_result$statistic,
    p_numeric = test_result$p,
    `p-value` = format_s6_p(test_result$p),
    Note = test_result$note,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
}

tableS6_numeric <- dplyr::bind_rows(s6_rows)

# Publication-ready version: omit the internal numeric p-value column.
tableS6 <- tableS6_numeric[, setdiff(names(tableS6_numeric), "p_numeric"), drop = FALSE]

# Group-level total N after eligibility filtering and deduplication.
tableS6_group_counts <- data.frame(
  Group = c(
    "Included (abuse_num = 1)",
    "Excluded (abuse_num = 2)"
  ),
  N = c(
    sum(df_s6$abuse_num_numeric == 1, na.rm = TRUE),
    sum(df_s6$abuse_num_numeric == 2, na.rm = TRUE)
  ),
  stringsAsFactors = FALSE
)

# -----------------------------------------------------------------------------
# 7. Save outputs
# -----------------------------------------------------------------------------

tableS6_file <- file.path(
  OUTPUT_DIR,
  paste0("tableS6_abuse_num1_vs_2_", timestamp, ".csv")
)

tableS6_group_counts_file <- file.path(
  OUTPUT_DIR,
  paste0("tableS6_group_counts_abuse_num1_vs_2_", timestamp, ".csv")
)

tableS6_numeric_file <- file.path(
  OUTPUT_DIR,
  paste0("tableS6_numeric_audit_abuse_num1_vs_2_", timestamp, ".csv")
)

write.csv(
  tableS6,
  tableS6_file,
  row.names = FALSE,
  na = "",
  fileEncoding = "UTF-8"
)

write.csv(
  tableS6_group_counts,
  tableS6_group_counts_file,
  row.names = FALSE,
  na = "",
  fileEncoding = "UTF-8"
)

write.csv(
  tableS6_numeric,
  tableS6_numeric_file,
  row.names = FALSE,
  na = "",
  fileEncoding = "UTF-8"
)

message("Table S6 saved to: ", tableS6_file)
message("Table S6 group counts saved to: ", tableS6_group_counts_file)
message("Table S6 numeric audit file saved to: ", tableS6_numeric_file)

print(tableS6_group_counts)
# print(tableS6)

cat(
  "Perm_DMFT non-missing N:",
  sum(!is.na(df$Perm_DMFT)),
  "\n"
)

cat(
  "Perm_DMFT non-missing & >=1 permanent tooth N:",
  sum(
    !is.na(df$Perm_DMFT) &
    !is.na(df$Perm_total_teeth) &
    df$Perm_total_teeth > 0
  ),
  "\n"
)

cat(
  "Perm_DMFT non-missing but 0 permanent teeth N:",
  sum(
    !is.na(df$Perm_DMFT) &
    !is.na(df$Perm_total_teeth) &
    df$Perm_total_teeth == 0
  ),
  "\n"
)


df_perm0 <- df %>%
  filter(
    !is.na(Perm_DMFT),
    !is.na(Perm_total_teeth),
    Perm_total_teeth == 0
  )

# 件数
nrow(df_perm0)

# 確認したい変数
df_perm0 %>%
  select(
    No_All,
    age_year,
    sex,
    abuse,
    Perm_DMFT,
    Perm_total_teeth,
    Baby_DMFT,
    Baby_total_teeth,
    DMFT_Index
  )
df_perm0 %>%
  select(
    No_All,
    age_year,
    Perm_DMFT,
    Perm_total_teeth,
    all_of(perm_cols)
  )
