# ============================================================================
# Machine-learning analysis for oral-health and abuse-related classifications
#
# 入力:
#   data/analysisData_20260211_AllData_cleaned.csv
#
# 目的:
#   1) Main CAN types vs Other consultation reasons
#   2) Neglect vs Non-neglect among main CAN types
#
# 手法:
#   - Logistic regression: 解釈しやすいベースライン
#   - Elastic-net logistic regression: 相関の強い歯科指標を正則化して扱う
#   - Random forest: 非線形・交互作用を拾う木系モデル
#   - Decision tree: 単純なルールとして見やすい補助モデル
#
# 出力:
#   result/ml_oral_health_abuse_YYYYMMDD/
# ============================================================================


# -----------------------------
# 0. Paths, packages, and utilities
# -----------------------------
# このブロックでは、入力CSVと出力先を指定し、評価指標(AUC、感度、特異度など)
# を計算するための補助関数を定義する。

cmd_args <- commandArgs(trailingOnly = FALSE)
file_arg <- cmd_args[grepl("^--file=", cmd_args)]
if (length(file_arg) > 0) {
  SCRIPT_PATH <- normalizePath(sub("^--file=", "", file_arg[1]), mustWork = FALSE)
  SCRIPT_DIR <- dirname(SCRIPT_PATH)
  BASE_DIR <- normalizePath(file.path(SCRIPT_DIR, ".."), mustWork = FALSE)
} else {
  BASE_DIR <- getwd()
  if (basename(BASE_DIR) == "code") {
    BASE_DIR <- normalizePath(file.path(BASE_DIR, ".."), mustWork = FALSE)
  }
}

DATA_PATH <- file.path(BASE_DIR, "data", "analysisData_20260211_AllData_cleaned.csv")
OUT_DIR <- file.path(BASE_DIR, "result", paste0("ml_oral_health_abuse_", format(Sys.Date(), "%Y%m%d")))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

set.seed(20260512)

has_glmnet <- requireNamespace("glmnet", quietly = TRUE)
has_randomForest <- requireNamespace("randomForest", quietly = TRUE)
has_rpart <- requireNamespace("rpart", quietly = TRUE)

format_p <- function(p) {
  if (length(p) == 0 || is.na(p) || !is.finite(p)) return("N/A")
  if (p < 0.001) return("<0.001")
  sprintf("%.3f", p)
}

safe_num <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

rank_auc <- function(y, score) {
  y <- as.integer(y)
  keep <- !is.na(y) & !is.na(score)
  y <- y[keep]
  score <- score[keep]
  n_pos <- sum(y == 1)
  n_neg <- sum(y == 0)
  if (n_pos == 0 || n_neg == 0) return(NA_real_)
  r <- rank(score, ties.method = "average")
  (sum(r[y == 1]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
}

class_weights <- function(y) {
  y <- as.integer(y)
  n <- length(y)
  n_pos <- sum(y == 1)
  n_neg <- sum(y == 0)
  w <- rep(1, n)
  if (n_pos > 0 && n_neg > 0) {
    w[y == 1] <- n / (2 * n_pos)
    w[y == 0] <- n / (2 * n_neg)
  }
  w
}

choose_threshold <- function(y, prob) {
  keep <- !is.na(y) & !is.na(prob)
  y <- as.integer(y[keep])
  prob <- prob[keep]
  if (length(unique(y)) < 2) return(0.5)
  cuts <- unique(as.numeric(quantile(prob, probs = seq(0.02, 0.98, by = 0.01), na.rm = TRUE)))
  if (length(cuts) == 0) return(0.5)
  best <- data.frame(threshold = cuts, score = NA_real_)
  for (i in seq_along(cuts)) {
    pred <- as.integer(prob >= cuts[i])
    tp <- sum(pred == 1 & y == 1)
    tn <- sum(pred == 0 & y == 0)
    fp <- sum(pred == 1 & y == 0)
    fn <- sum(pred == 0 & y == 1)
    sens <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
    spec <- ifelse(tn + fp > 0, tn / (tn + fp), NA_real_)
    best$score[i] <- sens + spec - 1
  }
  best$threshold[which.max(best$score)]
}

metrics_at_threshold <- function(y, prob, threshold) {
  keep <- !is.na(y) & !is.na(prob)
  y <- as.integer(y[keep])
  prob <- prob[keep]
  pred <- as.integer(prob >= threshold)
  tp <- sum(pred == 1 & y == 1)
  tn <- sum(pred == 0 & y == 0)
  fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1)
  sens <- ifelse(tp + fn > 0, tp / (tp + fn), NA_real_)
  spec <- ifelse(tn + fp > 0, tn / (tn + fp), NA_real_)
  ppv <- ifelse(tp + fp > 0, tp / (tp + fp), NA_real_)
  npv <- ifelse(tn + fn > 0, tn / (tn + fn), NA_real_)
  acc <- ifelse(length(y) > 0, (tp + tn) / length(y), NA_real_)
  bal_acc <- mean(c(sens, spec), na.rm = TRUE)
  data.frame(
    AUC = rank_auc(y, prob),
    Accuracy = acc,
    Balanced_Accuracy = bal_acc,
    Sensitivity = sens,
    Specificity = spec,
    PPV = ppv,
    NPV = npv,
    Threshold = threshold,
    TP = tp,
    TN = tn,
    FP = fp,
    FN = fn,
    stringsAsFactors = FALSE
  )
}

stratified_split <- function(y, train_prop = 0.7) {
  y <- as.integer(y)
  train_idx <- integer(0)
  for (cls in sort(unique(y[!is.na(y)]))) {
    idx <- which(y == cls)
    n_train <- floor(length(idx) * train_prop)
    train_idx <- c(train_idx, sample(idx, n_train))
  }
  sort(train_idx)
}

make_model_matrix <- function(df, feature_cols, train_idx) {
  x <- df[, feature_cols, drop = FALSE]
  for (nm in names(x)) {
    if (is.numeric(x[[nm]]) || is.integer(x[[nm]])) {
      med <- median(x[[nm]][train_idx], na.rm = TRUE)
      if (is.na(med) || !is.finite(med)) med <- 0
      x[[nm]][is.na(x[[nm]])] <- med
    } else {
      x[[nm]] <- as.character(x[[nm]])
      x[[nm]][is.na(x[[nm]]) | x[[nm]] == ""] <- "Missing"
      x[[nm]] <- factor(x[[nm]])
    }
  }
  mm <- model.matrix(~ . - 1, data = x)
  colnames(mm) <- make.names(colnames(mm), unique = TRUE)
  train_var <- apply(mm[train_idx, , drop = FALSE], 2, var)
  mm <- mm[, !is.na(train_var) & train_var > 0, drop = FALSE]
  mm
}

write_nonempty_csv <- function(x, path) {
  if (is.null(x) || nrow(x) == 0) {
    write.csv(data.frame(Note = "No rows generated."), path, row.names = FALSE)
  } else {
    write.csv(x, path, row.names = FALSE)
  }
}


# -----------------------------
# 1. Data loading and recoding
# -----------------------------
# このブロックでは、解析対象CSVを読み込み、虐待分類、日付、年齢、歯科所見の
# カテゴリを機械学習で扱いやすい形へ変換する。

if (!file.exists(DATA_PATH)) {
  stop("Data file not found: ", DATA_PATH)
}

data0 <- read.csv(DATA_PATH, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
message("Loaded: ", DATA_PATH)
message("Rows x columns: ", nrow(data0), " x ", ncol(data0))

if ("date" %in% names(data0)) {
  data0$date <- as.Date(data0$date)
  data0$exam_year <- as.integer(format(data0$date, "%Y"))
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
}

recode_factor <- function(x, mapping, levels_out) {
  x <- as.character(x)
  for (code in names(mapping)) {
    x[x == code] <- mapping[[code]]
  }
  factor(x, levels = levels_out)
}

if ("occlusalRelationship" %in% names(data0)) {
  data0$occlusalRelationship <- recode_factor(
    data0$occlusalRelationship,
    c("1" = "Normal Occlusion", "2" = "Crowding", "3" = "Anterior Crossbite",
      "4" = "Open Bite", "5" = "Maxillary Protrusion", "6" = "Crossbite", "7" = "Others"),
    c("Normal Occlusion", "Crowding", "Anterior Crossbite", "Open Bite",
      "Maxillary Protrusion", "Crossbite", "Others")
  )
}

if ("needTOBEtreated" %in% names(data0)) {
  data0$needTOBEtreated <- recode_factor(
    data0$needTOBEtreated,
    c("1" = "No Treatment Required", "2" = "Treatment Required"),
    c("No Treatment Required", "Treatment Required")
  )
}

if ("emergency" %in% names(data0)) {
  data0$emergency <- recode_factor(
    data0$emergency,
    c("1" = "Urgent Treatment Required"),
    c("Urgent Treatment Required")
  )
}

if ("gingivitis" %in% names(data0)) {
  data0$gingivitis <- recode_factor(
    data0$gingivitis,
    c("1" = "No Gingivitis", "2" = "Gingivitis"),
    c("No Gingivitis", "Gingivitis")
  )
}

if ("OralCleanStatus" %in% names(data0)) {
  data0$OralCleanStatus <- recode_factor(
    data0$OralCleanStatus,
    c("1" = "Poor", "2" = "Fair", "3" = "Good"),
    c("Poor", "Fair", "Good")
  )
}

if ("habits" %in% names(data0)) {
  data0$habits <- recode_factor(
    data0$habits,
    c("1" = "None", "2" = "Digit Sucking", "3" = "Nail biting",
      "4" = "Tongue Thrusting", "5" = "Smoking", "6" = "Others"),
    c("None", "Digit Sucking", "Nail biting", "Tongue Thrusting", "Smoking", "Others")
  )
}

for (nm in c("age_year", "age_month", "age", "emergencyInMonths")) {
  if (nm %in% names(data0)) data0[[nm]] <- safe_num(data0[[nm]])
}


# -----------------------------
# 2. Feature engineering from tooth-level codes
# -----------------------------
# このブロックでは、歯単位コードからdmft/DMFT、未処置う蝕、処置歯、
# 健全歯率、Care index、未処置う蝕率などの機械学習特徴量を作る。

perm_teeth_cols <- c(
  "U17","U16","U15","U14","U13","U12","U11","U21","U22","U23","U24","U25","U26","U27",
  "L37","L36","L35","L34","L33","L32","L31","L41","L42","L43","L44","L45","L46","L47"
)
primary_teeth_cols <- c(
  "u55","u54","u53","u52","u51","u61","u62","u63","u64","u65",
  "l75","l74","l73","l72","l71","l81","l82","l83","l84","l85"
)

perm_cols <- perm_teeth_cols[perm_teeth_cols %in% names(data0)]
primary_cols <- primary_teeth_cols[primary_teeth_cols %in% names(data0)]
for (tc in c(perm_cols, primary_cols)) {
  data0[[tc]] <- safe_num(data0[[tc]])
}

if (length(perm_cols) > 0) {
  pm <- data0[, perm_cols, drop = FALSE]
  pm_all_missing <- rowSums(!is.na(pm)) == 0
  data0$Perm_D <- rowSums(pm == 3, na.rm = TRUE)
  data0$Perm_M <- rowSums(pm == 4, na.rm = TRUE)
  data0$Perm_F <- rowSums(pm == 1, na.rm = TRUE)
  data0$Perm_C0 <- rowSums(pm == 2, na.rm = TRUE)
  data0$Perm_sound <- rowSums(pm == 0, na.rm = TRUE)
  data0$Perm_trauma <- rowSums(pm == 7, na.rm = TRUE)
  data0$Perm_congenital_missing <- rowSums(pm == 6, na.rm = TRUE)
  data0$Perm_recorded_teeth <- rowSums(!is.na(pm) & !(pm %in% c(-1, 6)), na.rm = TRUE)
  data0$Perm_DMFT <- data0$Perm_D + data0$Perm_M + data0$Perm_F
  for (v in c("Perm_D","Perm_M","Perm_F","Perm_C0","Perm_sound","Perm_trauma",
              "Perm_congenital_missing","Perm_recorded_teeth","Perm_DMFT")) {
    data0[[v]][pm_all_missing] <- NA_real_
  }
}

if (length(primary_cols) > 0) {
  bm <- data0[, primary_cols, drop = FALSE]
  bm_all_missing <- rowSums(!is.na(bm)) == 0
  data0$Primary_d <- rowSums(bm == 3, na.rm = TRUE)
  data0$Primary_m <- rowSums(bm == 4, na.rm = TRUE)
  data0$Primary_f <- rowSums(bm == 1, na.rm = TRUE)
  data0$Primary_C0 <- rowSums(bm == 2, na.rm = TRUE)
  data0$Primary_sound <- rowSums(bm == 0, na.rm = TRUE)
  data0$Primary_trauma <- rowSums(bm == 7, na.rm = TRUE)
  data0$Primary_retained <- rowSums(bm == 8, na.rm = TRUE)
  data0$Primary_fused <- rowSums(bm == 9, na.rm = TRUE)
  data0$Primary_congenital_missing <- rowSums(bm == 6, na.rm = TRUE)
  data0$Primary_recorded_teeth <- rowSums(!is.na(bm) & !(bm %in% c(-1, 6)), na.rm = TRUE)
  data0$Primary_dmft <- data0$Primary_d + data0$Primary_m + data0$Primary_f
  for (v in c("Primary_d","Primary_m","Primary_f","Primary_C0","Primary_sound",
              "Primary_trauma","Primary_retained","Primary_fused",
              "Primary_congenital_missing","Primary_recorded_teeth","Primary_dmft")) {
    data0[[v]][bm_all_missing] <- NA_real_
  }
}

if (!("Perm_DMFT" %in% names(data0))) data0$Perm_DMFT <- NA_real_
if (!("Primary_dmft" %in% names(data0))) data0$Primary_dmft <- NA_real_
data0$Total_dmft_DMFT <- ifelse(is.na(data0$Perm_DMFT), 0, data0$Perm_DMFT) +
  ifelse(is.na(data0$Primary_dmft), 0, data0$Primary_dmft)
data0$Total_dmft_DMFT[is.na(data0$Perm_DMFT) & is.na(data0$Primary_dmft)] <- NA_real_

data0$Decayed_total <- ifelse(is.na(data0$Perm_D), 0, data0$Perm_D) +
  ifelse(is.na(data0$Primary_d), 0, data0$Primary_d)
data0$Decayed_total[is.na(data0$Perm_D) & is.na(data0$Primary_d)] <- NA_real_

data0$Missing_total <- ifelse(is.na(data0$Perm_M), 0, data0$Perm_M) +
  ifelse(is.na(data0$Primary_m), 0, data0$Primary_m)
data0$Missing_total[is.na(data0$Perm_M) & is.na(data0$Primary_m)] <- NA_real_

data0$Filled_total <- ifelse(is.na(data0$Perm_F), 0, data0$Perm_F) +
  ifelse(is.na(data0$Primary_f), 0, data0$Primary_f)
data0$Filled_total[is.na(data0$Perm_F) & is.na(data0$Primary_f)] <- NA_real_

data0$C0_total <- ifelse(is.na(data0$Perm_C0), 0, data0$Perm_C0) +
  ifelse(is.na(data0$Primary_C0), 0, data0$Primary_C0)
data0$C0_total[is.na(data0$Perm_C0) & is.na(data0$Primary_C0)] <- NA_real_

data0$Trauma_total <- ifelse(is.na(data0$Perm_trauma), 0, data0$Perm_trauma) +
  ifelse(is.na(data0$Primary_trauma), 0, data0$Primary_trauma)
data0$Trauma_total[is.na(data0$Perm_trauma) & is.na(data0$Primary_trauma)] <- NA_real_

data0$Congenital_missing_total <- ifelse(is.na(data0$Perm_congenital_missing), 0, data0$Perm_congenital_missing) +
  ifelse(is.na(data0$Primary_congenital_missing), 0, data0$Primary_congenital_missing)
data0$Congenital_missing_total[is.na(data0$Perm_congenital_missing) & is.na(data0$Primary_congenital_missing)] <- NA_real_

data0$Recorded_teeth_total <- ifelse(is.na(data0$Perm_recorded_teeth), 0, data0$Perm_recorded_teeth) +
  ifelse(is.na(data0$Primary_recorded_teeth), 0, data0$Primary_recorded_teeth)
data0$Recorded_teeth_total[is.na(data0$Perm_recorded_teeth) & is.na(data0$Primary_recorded_teeth)] <- NA_real_

data0$Sound_total <- ifelse(is.na(data0$Perm_sound), 0, data0$Perm_sound) +
  ifelse(is.na(data0$Primary_sound), 0, data0$Primary_sound)
data0$Sound_total[is.na(data0$Perm_sound) & is.na(data0$Primary_sound)] <- NA_real_

data0$Healthy_rate <- data0$Sound_total / data0$Recorded_teeth_total * 100
data0$Healthy_rate[!is.finite(data0$Healthy_rate) | data0$Recorded_teeth_total <= 0] <- NA_real_

data0$Care_index <- data0$Filled_total / data0$Total_dmft_DMFT * 100
data0$Care_index[!is.finite(data0$Care_index) | data0$Total_dmft_DMFT <= 0] <- NA_real_

data0$Untreated_caries_rate <- data0$Decayed_total / data0$Total_dmft_DMFT * 100
data0$Untreated_caries_rate[!is.finite(data0$Untreated_caries_rate) | data0$Total_dmft_DMFT <= 0] <- NA_real_

data0$Primary_caries_present <- ifelse(!is.na(data0$Primary_d), as.integer(data0$Primary_d > 0), NA)
data0$Permanent_caries_present <- ifelse(!is.na(data0$Perm_D), as.integer(data0$Perm_D > 0), NA)
data0$Any_caries_experience <- ifelse(!is.na(data0$Total_dmft_DMFT), as.integer(data0$Total_dmft_DMFT > 0), NA)
data0$Untreated_caries_present <- ifelse(!is.na(data0$Decayed_total), as.integer(data0$Decayed_total > 0), NA)
data0$Permanent_fillings_present <- ifelse(!is.na(data0$Perm_F), as.integer(data0$Perm_F > 0), NA)
data0$Primary_fillings_present <- ifelse(!is.na(data0$Primary_f), as.integer(data0$Primary_f > 0), NA)
data0$Treatment_need <- ifelse(!is.na(data0$needTOBEtreated), as.integer(data0$needTOBEtreated == "Treatment Required"), NA)
data0$Urgent_treatment <- ifelse(!is.na(data0$emergency), as.integer(data0$emergency == "Urgent Treatment Required"), 0)
data0$Gingivitis_present <- ifelse(!is.na(data0$gingivitis), as.integer(data0$gingivitis == "Gingivitis"), NA)
data0$Poor_oral_hygiene <- ifelse(!is.na(data0$OralCleanStatus), as.integer(data0$OralCleanStatus == "Poor"), NA)
data0$Any_malocclusion <- ifelse(!is.na(data0$occlusalRelationship), as.integer(data0$occlusalRelationship != "Normal Occlusion"), NA)
data0$Oral_habit_present <- ifelse(!is.na(data0$habits), as.integer(data0$habits != "None"), NA)
data0$Dental_trauma_present <- ifelse(!is.na(data0$Trauma_total), as.integer(data0$Trauma_total > 0), NA)
data0$Retained_deciduous_present <- ifelse(!is.na(data0$Primary_retained), as.integer(data0$Primary_retained > 0), NA)
data0$Congenital_missing_present <- ifelse(!is.na(data0$Congenital_missing_total), as.integer(data0$Congenital_missing_total > 0), NA)


# -----------------------------
# 3. Outcomes and feature sets
# -----------------------------
# このブロックでは、予測したいアウトカムを2種類作る。
# Task A: 主要CAN群 vs その他相談理由群
# Task B: 主要CAN群の中でNeglect vs 非Neglect

main_can_types <- c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")
other_reason_types <- c("Delinquency", "Parenting Difficulties", "Others")

data0$task_main_can <- NA_integer_
data0$task_main_can[data0$abuse %in% main_can_types] <- 1
data0$task_main_can[data0$abuse %in% other_reason_types] <- 0

data0$task_neglect <- NA_integer_
data0$task_neglect[data0$abuse %in% main_can_types] <- as.integer(data0$abuse[data0$abuse %in% main_can_types] == "Neglect")

derived_oral_features <- c(
  "Perm_D", "Perm_M", "Perm_F", "Perm_C0", "Perm_sound", "Perm_trauma",
  "Perm_congenital_missing", "Perm_recorded_teeth", "Perm_DMFT",
  "Primary_d", "Primary_m", "Primary_f", "Primary_C0", "Primary_sound",
  "Primary_trauma", "Primary_retained", "Primary_fused",
  "Primary_congenital_missing", "Primary_recorded_teeth", "Primary_dmft",
  "Total_dmft_DMFT", "Decayed_total", "Missing_total", "Filled_total",
  "C0_total", "Trauma_total", "Congenital_missing_total",
  "Recorded_teeth_total", "Sound_total", "Healthy_rate", "Care_index",
  "Untreated_caries_rate", "Primary_caries_present",
  "Permanent_caries_present", "Any_caries_experience",
  "Untreated_caries_present", "Permanent_fillings_present",
  "Primary_fillings_present", "Treatment_need", "Urgent_treatment",
  "Gingivitis_present", "Poor_oral_hygiene", "Any_malocclusion",
  "Oral_habit_present", "Dental_trauma_present", "Retained_deciduous_present",
  "Congenital_missing_present"
)

clinical_categorical_features <- c(
  "needTOBEtreated", "emergency", "emergencyInMonths", "gingivitis",
  "occlusalRelationship", "habits", "OralCleanStatus", "Orthodontics",
  "dentists", "dental_hygienist"
)

behavior_features <- c(
  "wake_up", "breakfast", "morning_brushing", "school", "bedtime",
  "night_brushing", "TV", "game", "meal", "extra_lesson"
)

demographic_features <- c("age_year", "age_month", "sex", "CGC", "exam_year")

oral_only_features <- unique(c(derived_oral_features, clinical_categorical_features, behavior_features))
oral_only_features <- oral_only_features[oral_only_features %in% names(data0)]

oral_plus_demo_features <- unique(c(oral_only_features, demographic_features))
oral_plus_demo_features <- oral_plus_demo_features[oral_plus_demo_features %in% names(data0)]


# -----------------------------
# 4. Model runner
# -----------------------------
# このブロックでは、同じtrain/test分割に対して複数の機械学習手法を実行し、
# AUC、正確度、balanced accuracy、感度、特異度を比較する。

run_binary_ml_task <- function(df, outcome_col, task_name, positive_label, feature_sets) {
  task_dir <- file.path(OUT_DIR, task_name)
  dir.create(task_dir, recursive = TRUE, showWarnings = FALSE)

  df_task <- df[!is.na(df[[outcome_col]]), , drop = FALSE]
  df_task[[outcome_col]] <- as.integer(df_task[[outcome_col]])
  y_all <- df_task[[outcome_col]]
  train_idx <- stratified_split(y_all, train_prop = 0.7)
  test_idx <- setdiff(seq_len(nrow(df_task)), train_idx)

  split_info <- data.frame(
    Task = task_name,
    Positive_Label = positive_label,
    N_total = nrow(df_task),
    N_train = length(train_idx),
    N_test = length(test_idx),
    Positive_total = sum(y_all == 1),
    Positive_train = sum(y_all[train_idx] == 1),
    Positive_test = sum(y_all[test_idx] == 1),
    Positive_rate_total = mean(y_all == 1),
    Positive_rate_train = mean(y_all[train_idx] == 1),
    Positive_rate_test = mean(y_all[test_idx] == 1),
    stringsAsFactors = FALSE
  )
  write.csv(split_info, file.path(task_dir, "split_info.csv"), row.names = FALSE)

  all_perf <- list()
  all_predictions <- list()

  for (feature_set_name in names(feature_sets)) {
    feature_cols <- feature_sets[[feature_set_name]]
    feature_cols <- feature_cols[feature_cols %in% names(df_task)]
    x_all <- make_model_matrix(df_task, feature_cols, train_idx)
    x_train <- x_all[train_idx, , drop = FALSE]
    x_test <- x_all[test_idx, , drop = FALSE]
    y_train <- y_all[train_idx]
    y_test <- y_all[test_idx]
    w_train <- class_weights(y_train)

    feature_dir <- file.path(task_dir, feature_set_name)
    dir.create(feature_dir, recursive = TRUE, showWarnings = FALSE)
    write.csv(
      data.frame(feature = colnames(x_train), stringsAsFactors = FALSE),
      file.path(feature_dir, "model_matrix_features.csv"),
      row.names = FALSE
    )

    # Logistic regression baseline.
    glm_df_train <- data.frame(y = y_train, x_train, check.names = FALSE)
    glm_df_test <- data.frame(x_test, check.names = FALSE)
    fit_glm <- try(
      suppressWarnings(glm(y ~ ., data = glm_df_train, family = binomial(), weights = w_train,
          control = glm.control(maxit = 100))),
      silent = TRUE
    )
    if (!inherits(fit_glm, "try-error")) {
      prob_train <- suppressWarnings(as.numeric(predict(fit_glm, newdata = glm_df_train, type = "response")))
      prob_test <- suppressWarnings(as.numeric(predict(fit_glm, newdata = glm_df_test, type = "response")))
      threshold <- choose_threshold(y_train, prob_train)
      perf <- metrics_at_threshold(y_test, prob_test, threshold)
      perf <- cbind(
        Task = task_name, Feature_Set = feature_set_name,
        Model = "Logistic regression", N_train = length(y_train),
        N_test = length(y_test), Positive_Label = positive_label, perf
      )
      all_perf[[length(all_perf) + 1]] <- perf

      glm_coef <- summary(fit_glm)$coefficients
      glm_coef <- data.frame(
        Feature = rownames(glm_coef),
        Estimate = glm_coef[, "Estimate"],
        Std_Error = glm_coef[, "Std. Error"],
        p_value = glm_coef[, "Pr(>|z|)"],
        p = vapply(glm_coef[, "Pr(>|z|)"], format_p, character(1)),
        row.names = NULL
      )
      glm_coef <- glm_coef[order(abs(glm_coef$Estimate), decreasing = TRUE), , drop = FALSE]
      write.csv(glm_coef, file.path(feature_dir, "logistic_coefficients.csv"), row.names = FALSE)

      all_predictions[[length(all_predictions) + 1]] <- data.frame(
        Task = task_name, Feature_Set = feature_set_name,
        Model = "Logistic regression", Row_Index = test_idx,
        Truth = y_test, Probability = prob_test,
        Prediction = as.integer(prob_test >= threshold),
        stringsAsFactors = FALSE
      )
    }

    # Elastic-net logistic regression.
    if (has_glmnet) {
      fit_en <- try(
        glmnet::cv.glmnet(
          x = x_train, y = y_train, family = "binomial",
          alpha = 0.5, nfolds = 5, type.measure = "auc",
          weights = w_train, standardize = TRUE
        ),
        silent = TRUE
      )
      if (!inherits(fit_en, "try-error")) {
        prob_train <- as.numeric(predict(fit_en, newx = x_train, s = "lambda.1se", type = "response"))
        prob_test <- as.numeric(predict(fit_en, newx = x_test, s = "lambda.1se", type = "response"))
        threshold <- choose_threshold(y_train, prob_train)
        perf <- metrics_at_threshold(y_test, prob_test, threshold)
        perf <- cbind(
          Task = task_name, Feature_Set = feature_set_name,
          Model = "Elastic-net logistic", N_train = length(y_train),
          N_test = length(y_test), Positive_Label = positive_label, perf
        )
        all_perf[[length(all_perf) + 1]] <- perf

        coef_mat <- as.matrix(coef(fit_en, s = "lambda.1se"))
        coef_df <- data.frame(
          Feature = rownames(coef_mat),
          Coefficient = as.numeric(coef_mat[, 1]),
          row.names = NULL
        )
        coef_df <- coef_df[coef_df$Coefficient != 0, , drop = FALSE]
        coef_df <- coef_df[order(abs(coef_df$Coefficient), decreasing = TRUE), , drop = FALSE]
        coef_df$lambda_1se <- fit_en$lambda.1se
        coef_df$lambda_min <- fit_en$lambda.min
        write_nonempty_csv(coef_df, file.path(feature_dir, "elastic_net_nonzero_coefficients.csv"))

        all_predictions[[length(all_predictions) + 1]] <- data.frame(
          Task = task_name, Feature_Set = feature_set_name,
          Model = "Elastic-net logistic", Row_Index = test_idx,
          Truth = y_test, Probability = prob_test,
          Prediction = as.integer(prob_test >= threshold),
          stringsAsFactors = FALSE
        )
      }
    }

    # Random forest.
    if (has_randomForest) {
      y_train_fac <- factor(ifelse(y_train == 1, "Positive", "Negative"), levels = c("Negative", "Positive"))
      rf_classwt <- c(
        Negative = length(y_train) / (2 * sum(y_train == 0)),
        Positive = length(y_train) / (2 * sum(y_train == 1))
      )
      fit_rf <- try(
        randomForest::randomForest(
          x = x_train, y = y_train_fac, ntree = 500,
          mtry = max(1, floor(sqrt(ncol(x_train)))),
          importance = TRUE, classwt = rf_classwt
        ),
        silent = TRUE
      )
      if (!inherits(fit_rf, "try-error")) {
        prob_train <- as.numeric(predict(fit_rf, x_train, type = "prob")[, "Positive"])
        prob_test <- as.numeric(predict(fit_rf, x_test, type = "prob")[, "Positive"])
        threshold <- choose_threshold(y_train, prob_train)
        perf <- metrics_at_threshold(y_test, prob_test, threshold)
        perf <- cbind(
          Task = task_name, Feature_Set = feature_set_name,
          Model = "Random forest", N_train = length(y_train),
          N_test = length(y_test), Positive_Label = positive_label, perf
        )
        all_perf[[length(all_perf) + 1]] <- perf

        imp <- randomForest::importance(fit_rf)
        imp_df <- data.frame(Feature = rownames(imp), imp, row.names = NULL, check.names = FALSE)
        sort_col <- if ("MeanDecreaseGini" %in% names(imp_df)) "MeanDecreaseGini" else names(imp_df)[2]
        imp_df <- imp_df[order(imp_df[[sort_col]], decreasing = TRUE), , drop = FALSE]
        write.csv(imp_df, file.path(feature_dir, "random_forest_importance.csv"), row.names = FALSE)

        all_predictions[[length(all_predictions) + 1]] <- data.frame(
          Task = task_name, Feature_Set = feature_set_name,
          Model = "Random forest", Row_Index = test_idx,
          Truth = y_test, Probability = prob_test,
          Prediction = as.integer(prob_test >= threshold),
          stringsAsFactors = FALSE
        )
      }
    }

    # Decision tree.
    if (has_rpart) {
      tree_train <- data.frame(
        y = factor(ifelse(y_train == 1, "Positive", "Negative"), levels = c("Negative", "Positive")),
        x_train,
        check.names = FALSE
      )
      tree_test <- data.frame(x_test, check.names = FALSE)
      fit_tree <- try(
        rpart::rpart(
          y ~ ., data = tree_train, method = "class", weights = w_train,
          control = rpart::rpart.control(cp = 0.005, minbucket = 20, maxdepth = 5)
        ),
        silent = TRUE
      )
      if (!inherits(fit_tree, "try-error")) {
        prob_train <- as.numeric(predict(fit_tree, tree_train, type = "prob")[, "Positive"])
        prob_test <- as.numeric(predict(fit_tree, tree_test, type = "prob")[, "Positive"])
        threshold <- choose_threshold(y_train, prob_train)
        perf <- metrics_at_threshold(y_test, prob_test, threshold)
        perf <- cbind(
          Task = task_name, Feature_Set = feature_set_name,
          Model = "Decision tree", N_train = length(y_train),
          N_test = length(y_test), Positive_Label = positive_label, perf
        )
        all_perf[[length(all_perf) + 1]] <- perf

        tree_imp <- fit_tree$variable.importance
        if (!is.null(tree_imp)) {
          tree_imp_df <- data.frame(
            Feature = names(tree_imp),
            Importance = as.numeric(tree_imp),
            row.names = NULL
          )
          tree_imp_df <- tree_imp_df[order(tree_imp_df$Importance, decreasing = TRUE), , drop = FALSE]
          write.csv(tree_imp_df, file.path(feature_dir, "decision_tree_importance.csv"), row.names = FALSE)
        }
        capture.output(print(fit_tree), file = file.path(feature_dir, "decision_tree_print.txt"))

        all_predictions[[length(all_predictions) + 1]] <- data.frame(
          Task = task_name, Feature_Set = feature_set_name,
          Model = "Decision tree", Row_Index = test_idx,
          Truth = y_test, Probability = prob_test,
          Prediction = as.integer(prob_test >= threshold),
          stringsAsFactors = FALSE
        )
      }
    }
  }

  perf_all <- if (length(all_perf) > 0) do.call(rbind, all_perf) else data.frame()
  pred_all <- if (length(all_predictions) > 0) do.call(rbind, all_predictions) else data.frame()
  if (nrow(perf_all) > 0) {
    metric_cols <- c("AUC", "Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity", "PPV", "NPV", "Threshold")
    for (nm in metric_cols) perf_all[[nm]] <- round(as.numeric(perf_all[[nm]]), 4)
    perf_all <- perf_all[order(perf_all$AUC, perf_all$Balanced_Accuracy, decreasing = TRUE), , drop = FALSE]
  }
  write_nonempty_csv(perf_all, file.path(task_dir, "model_performance.csv"))
  write_nonempty_csv(pred_all, file.path(task_dir, "test_predictions.csv"))
  perf_all
}


# -----------------------------
# 5. Run tasks
# -----------------------------
# このブロックでは、2つのアウトカムに対して、口腔指標のみのモデルと
# 年齢・性別・施設・年度を加えたモデルを実行する。

feature_sets <- list(
  oral_only = oral_only_features,
  oral_plus_demographics = oral_plus_demo_features
)

performance_main_can <- run_binary_ml_task(
  df = data0,
  outcome_col = "task_main_can",
  task_name = "task_A_main_CAN_vs_other",
  positive_label = "Main CAN types",
  feature_sets = feature_sets
)

performance_neglect <- run_binary_ml_task(
  df = data0,
  outcome_col = "task_neglect",
  task_name = "task_B_neglect_vs_other_main_CAN",
  positive_label = "Neglect",
  feature_sets = feature_sets
)

performance_all <- rbind(performance_main_can, performance_neglect)
write_nonempty_csv(performance_all, file.path(OUT_DIR, "model_performance_all_tasks.csv"))


# -----------------------------
# 6. Analysis notes
# -----------------------------
# このブロックでは、機械学習結果を論文で使う際の注意点を保存する。

notes <- c(
  "Machine-learning analysis notes",
  "===============================",
  "",
  "Recommended interpretation:",
  "- Treat these models as exploratory prediction models, not diagnostic tools.",
  "- Compare oral_only with oral_plus_demographics to check whether prediction is driven mostly by age/sex/site/year.",
  "- Use AUC and balanced accuracy rather than accuracy alone because class balance differs by task.",
  "- Elastic-net coefficients are useful for parsimonious feature selection.",
  "- Random forest importance is useful for nonlinear screening, but it is not causal evidence.",
  "",
  "Implemented tasks:",
  "- Task A: Main CAN types vs Other consultation reasons.",
  "- Task B: Neglect vs non-neglect among main CAN types.",
  "",
  "Implemented models:",
  "- Logistic regression.",
  if (has_glmnet) "- Elastic-net logistic regression (glmnet)." else "- Elastic-net logistic regression skipped because glmnet is not installed.",
  if (has_randomForest) "- Random forest (randomForest)." else "- Random forest skipped because randomForest is not installed.",
  if (has_rpart) "- Decision tree (rpart)." else "- Decision tree skipped because rpart is not installed.",
  "",
  "Main output files:",
  "- model_performance_all_tasks.csv",
  "- task_A_main_CAN_vs_other/model_performance.csv",
  "- task_B_neglect_vs_other_main_CAN/model_performance.csv",
  "- Feature-specific coefficient and importance files under each task/feature-set directory."
)
writeLines(notes, file.path(OUT_DIR, "analysis_notes.txt"))

message("Machine-learning analysis complete. Output directory: ", OUT_DIR)
