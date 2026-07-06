# ============================================================================
# 論文「Oral health in children investigated by Social services on suspicion
# of child abuse and neglect」(Kvist et al., 2018) の解析手順を参考にしたRコード
#
# 入力データ:
#   data/analysisData_20260211_AllData_cleaned.csv
#
# 注意:
#   参照論文は「CAN疑いで調査された児」と「年齢・性別・歯科医院でマッチした対照群」
#   を比較しています。一方、このCSVには一般対照群が含まれていないため、本コードでは
#   同じ統計手順を以下の2つに置き換えて実行します。
#     1) 主要CAN 4分類(Physical/Neglect/Emotional/Sexual) vs その他相談理由
#     2) 主要CAN 4分類内での虐待種別間比較
#
# 出力:
#   result/paper_like_kvist2018_YYYYMMDD/
# ============================================================================


# -----------------------------
# 0. パスと小さな補助関数
# -----------------------------
# このブロックでは、解析に用いるCSVと結果出力フォルダーを指定し、
# p値や平均±SDなどを表形式に整えるための補助関数を定義する。

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

DATA_PATH <- file.path(BASE_DIR, "paper_analysis","data", "analysisData_20260211_AllData_cleaned.csv")
OUT_DIR <- file.path(BASE_DIR,"paper_analysis","result", paste0("paper_like_kvist2018_", format(Sys.Date(), "%Y%m%d")))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

format_p <- function(p) {
  if (length(p) == 0 || is.na(p) || !is.finite(p)) return("N/A")
  if (p < 0.001) return("<0.001")
  sprintf("%.3f", p)
}

mean_sd <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return("N/A")
  sprintf("%.2f ± %.2f", mean(x), sd(x))
}

median_iqr <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return("N/A")
  q <- quantile(x, c(0.25, 0.75), na.rm = TRUE, names = FALSE)
  sprintf("%.1f [%.1f-%.1f]", median(x), q[1], q[2])
}

safe_num <- function(x) {
  suppressWarnings(as.numeric(as.character(x)))
}

safe_t_p <- function(x, g) {
  keep <- !is.na(x) & !is.na(g)
  x <- x[keep]
  g <- droplevels(factor(g[keep]))
  if (length(unique(g)) != 2) return(NA_real_)
  if (length(unique(x[g == levels(g)[1]])) < 2 && length(unique(x[g == levels(g)[2]])) < 2) return(NA_real_)
  out <- try(t.test(x ~ g)$p.value, silent = TRUE)
  if (inherits(out, "try-error")) NA_real_ else out
}

safe_wilcox_p <- function(x, g) {
  keep <- !is.na(x) & !is.na(g)
  x <- x[keep]
  g <- droplevels(factor(g[keep]))
  if (length(unique(g)) != 2) return(NA_real_)
  out <- try(wilcox.test(x ~ g, exact = FALSE)$p.value, silent = TRUE)
  if (inherits(out, "try-error")) NA_real_ else out
}

safe_anova_p <- function(x, g) {
  keep <- !is.na(x) & !is.na(g)
  x <- x[keep]
  g <- droplevels(factor(g[keep]))
  if (length(unique(g)) < 2) return(NA_real_)
  out <- try(summary(aov(x ~ g))[[1]][["Pr(>F)"]][1], silent = TRUE)
  if (inherits(out, "try-error")) NA_real_ else out
}

safe_kruskal_p <- function(x, g) {
  keep <- !is.na(x) & !is.na(g)
  x <- x[keep]
  g <- droplevels(factor(g[keep]))
  if (length(unique(g)) < 2) return(NA_real_)
  out <- try(kruskal.test(x ~ g)$p.value, silent = TRUE)
  if (inherits(out, "try-error")) NA_real_ else out
}

safe_cat_test <- function(x, g) {
  keep <- !is.na(x) & !is.na(g)
  x <- x[keep]
  g <- droplevels(factor(g[keep]))
  x <- droplevels(factor(x))
  if (length(unique(g)) < 2 || length(unique(x)) < 2) {
    return(list(method = "N/A", p = NA_real_))
  }
  tab <- table(g, x)
  chi <- suppressWarnings(try(chisq.test(tab, correct = FALSE), silent = TRUE))
  if (!inherits(chi, "try-error") && all(chi$expected >= 5)) {
    return(list(method = "Chi-square", p = chi$p.value))
  }
  fisher <- try(fisher.test(tab), silent = TRUE)
  if (!inherits(fisher, "try-error")) {
    return(list(method = "Fisher exact", p = fisher$p.value))
  }
  fisher_sim <- try(fisher.test(tab, simulate.p.value = TRUE, B = 10000), silent = TRUE)
  if (!inherits(fisher_sim, "try-error")) {
    return(list(method = "Fisher exact simulated", p = fisher_sim$p.value))
  }
  if (!inherits(chi, "try-error")) {
    return(list(method = "Chi-square", p = chi$p.value))
  }
  list(method = "N/A", p = NA_real_)
}

logistic_or_table <- function(fit, model_name, label_lookup) {
  coef_tab <- summary(fit)$coefficients
  rows <- list()
  for (term in rownames(coef_tab)) {
    if (term == "(Intercept)") next
    beta <- coef_tab[term, "Estimate"]
    se <- coef_tab[term, "Std. Error"]
    p <- coef_tab[term, "Pr(>|z|)"]
    ci_low <- exp(beta - 1.96 * se)
    ci_high <- exp(beta + 1.96 * se)
    label <- ifelse(term %in% names(label_lookup), label_lookup[[term]], term)
    rows[[length(rows) + 1]] <- data.frame(
      Model = model_name,
      Term = term,
      Label = label,
      Beta = round(beta, 3),
      OR = round(exp(beta), 3),
      CI_95 = sprintf("%.3f-%.3f", ci_low, ci_high),
      p_value = p,
      p = format_p(p),
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0) data.frame() else do.call(rbind, rows)
}


# -----------------------------
# 1. データ読み込み
# -----------------------------
# このブロックでは、dataフォルダー内のクリーニング済みCSVを読み込み、
# 行数・列数と日付範囲を記録する。論文と同様に診療記録ベースの後方視的データとして扱う。

if (!file.exists(DATA_PATH)) {
  stop("Data file not found: ", DATA_PATH)
}

data0 <- read.csv(DATA_PATH, stringsAsFactors = FALSE, na.strings = c("", "NA", "NaN"))
message("Loaded: ", DATA_PATH)
message("Rows x columns: ", nrow(data0), " x ", ncol(data0))
# 2480 x 81

# -----------------------------
# 2. カテゴリ変数の再コード化
# -----------------------------
# このブロックでは、ユーザー指定のcodingに従い、虐待種別、咬合、治療必要性、
# 歯肉炎、口腔清掃状態、習癖を解析しやすいfactorへ変換する。

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


# -----------------------------
# 3. 歯単位コードから口腔アウトカムを作成
# -----------------------------
# このブロックでは、各歯のコードを用いて、論文の「乳歯・永久歯のう蝕、
# 喪失、処置歯、総数」に相当する変数を作る。
# 歯コード: 未萌出=-1, 健全=0, 処置歯=1, C0=2, C=3, 喪失歯=4,
# その他/過剰歯=5, 先天性欠損=6, 歯牙破折=7, 乳歯晩期残存=8, 癒合歯=9。

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
  data0$Perm_D <- rowSums(pm == 3, na.rm = TRUE) # 未処置う蝕歯数
  data0$Perm_M <- rowSums(pm == 4, na.rm = TRUE) # 喪失歯数
  data0$Perm_F <- rowSums(pm == 1, na.rm = TRUE) # 処置歯数
  data0$Perm_C0 <- rowSums(pm == 2, na.rm = TRUE) # C0歯数
  data0$Perm_sound <- rowSums(pm == 0, na.rm = TRUE) # 健全歯数
  data0$Perm_trauma <- rowSums(pm == 7, na.rm = TRUE) # 歯牙破折歯数
  data0$Perm_congenital_missing <- rowSums(pm == 6, na.rm = TRUE) # 先天性欠損歯数
  data0$Perm_recorded_teeth <- rowSums(!is.na(pm) & !(pm %in% c(-1, 6)), na.rm = TRUE) # 記録された歯の総数(記録なし、未萌出、先天性欠損歯を除く)
  data0$Perm_DMFT <- data0$Perm_D + data0$Perm_M + data0$Perm_F # 永久歯のDMFT
  for (v in c("Perm_D","Perm_M","Perm_F","Perm_C0","Perm_sound","Perm_trauma","Perm_congenital_missing","Perm_recorded_teeth","Perm_DMFT")) {
    data0[[v]][pm_all_missing] <- NA_real_ # 永久歯データがすべてNAの場合、NAとして処理
  }
} else {
  data0$Perm_D <- data0$Perm_M <- data0$Perm_F <- data0$Perm_C0 <- NA_real_
  data0$Perm_sound <- data0$Perm_trauma <- data0$Perm_congenital_missing <- NA_real_
  data0$Perm_recorded_teeth <- data0$Perm_DMFT <- NA_real_
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
  data0$Primary_retained <- rowSums(bm == 8, na.rm = TRUE) # 晩期残存歯数
  data0$Primary_fused <- rowSums(bm == 9, na.rm = TRUE) # 癒合歯数
  data0$Primary_congenital_missing <- rowSums(bm == 6, na.rm = TRUE) # 先天性欠損歯数
  data0$Primary_recorded_teeth <- rowSums(!is.na(bm) & !(bm %in% c(-1, 6)), na.rm = TRUE)
  data0$Primary_dmft <- data0$Primary_d + data0$Primary_m + data0$Primary_f
  for (v in c("Primary_d","Primary_m","Primary_f","Primary_C0","Primary_sound","Primary_trauma","Primary_retained","Primary_fused","Primary_congenital_missing","Primary_recorded_teeth","Primary_dmft")) {
    data0[[v]][bm_all_missing] <- NA_real_
  }
} else {
  data0$Primary_d <- data0$Primary_m <- data0$Primary_f <- data0$Primary_C0 <- NA_real_
  data0$Primary_sound <- data0$Primary_trauma <- data0$Primary_retained <- data0$Primary_fused <- NA_real_
  data0$Primary_congenital_missing <- data0$Primary_recorded_teeth <- data0$Primary_dmft <- NA_real_
}

both_dmft_missing <- is.na(data0$Perm_DMFT) & is.na(data0$Primary_dmft)
data0$Total_dmft_DMFT <- ifelse(is.na(data0$Perm_DMFT), 0, data0$Perm_DMFT) + ifelse(is.na(data0$Primary_dmft), 0, data0$Primary_dmft)
data0$Total_dmft_DMFT[both_dmft_missing] <- NA_real_

both_decayed_missing <- is.na(data0$Perm_D) & is.na(data0$Primary_d)
data0$Decayed_total <- ifelse(is.na(data0$Perm_D), 0, data0$Perm_D) + ifelse(is.na(data0$Primary_d), 0, data0$Primary_d)
data0$Decayed_total[both_decayed_missing] <- NA_real_

both_missing_missing <- is.na(data0$Perm_M) & is.na(data0$Primary_m)
data0$Missing_total <- ifelse(is.na(data0$Perm_M), 0, data0$Perm_M) + ifelse(is.na(data0$Primary_m), 0, data0$Primary_m)
data0$Missing_total[both_missing_missing] <- NA_real_

both_filled_missing <- is.na(data0$Perm_F) & is.na(data0$Primary_f)
data0$Filled_total <- ifelse(is.na(data0$Perm_F), 0, data0$Perm_F) + ifelse(is.na(data0$Primary_f), 0, data0$Primary_f)
data0$Filled_total[both_filled_missing] <- NA_real_

data0$C0_total <- ifelse(is.na(data0$Perm_C0), 0, data0$Perm_C0) + ifelse(is.na(data0$Primary_C0), 0, data0$Primary_C0)
data0$C0_total[is.na(data0$Perm_C0) & is.na(data0$Primary_C0)] <- NA_real_

data0$Trauma_total <- ifelse(is.na(data0$Perm_trauma), 0, data0$Perm_trauma) + ifelse(is.na(data0$Primary_trauma), 0, data0$Primary_trauma)
data0$Trauma_total[is.na(data0$Perm_trauma) & is.na(data0$Primary_trauma)] <- NA_real_

data0$Congenital_missing_total <- ifelse(is.na(data0$Perm_congenital_missing), 0, data0$Perm_congenital_missing) + ifelse(is.na(data0$Primary_congenital_missing), 0, data0$Primary_congenital_missing)
data0$Congenital_missing_total[is.na(data0$Perm_congenital_missing) & is.na(data0$Primary_congenital_missing)] <- NA_real_

data0$Recorded_teeth_total <- ifelse(is.na(data0$Perm_recorded_teeth), 0, data0$Perm_recorded_teeth) + ifelse(is.na(data0$Primary_recorded_teeth), 0, data0$Primary_recorded_teeth)
data0$Recorded_teeth_total[is.na(data0$Perm_recorded_teeth) & is.na(data0$Primary_recorded_teeth)] <- NA_real_

data0$Sound_total <- ifelse(is.na(data0$Perm_sound), 0, data0$Perm_sound) + ifelse(is.na(data0$Primary_sound), 0, data0$Primary_sound)
data0$Sound_total[is.na(data0$Perm_sound) & is.na(data0$Primary_sound)] <- NA_real_

data0$Healthy_rate <- data0$Sound_total / data0$Recorded_teeth_total * 100
data0$Healthy_rate[!is.finite(data0$Healthy_rate) | data0$Recorded_teeth_total <= 0] <- NA_real_

data0$Care_index <- data0$Filled_total / data0$Total_dmft_DMFT * 100
data0$Care_index[!is.finite(data0$Care_index) | data0$Total_dmft_DMFT <= 0] <- NA_real_

data0$Untreated_caries_rate <- data0$Decayed_total / data0$Total_dmft_DMFT * 100
data0$Untreated_caries_rate[!is.finite(data0$Untreated_caries_rate) | data0$Total_dmft_DMFT <= 0] <- NA_real_


# -----------------------------
# 4. 論文の二群比較に相当する解析群を作成
# -----------------------------
# このブロックでは、論文の「CAN疑い群 vs 対照群」に相当する比較軸として、
# 本データ内の主要CAN 4分類 vs その他相談理由を作成する。
# これは一般人口対照ではないため、結果解釈では「その他相談理由との比較」として扱う。

main_can_types <- c("Physical Abuse", "Neglect", "Emotional Abuse", "Sexual Abuse")
other_reason_types <- c("Delinquency", "Parenting Difficulties", "Others")

data0$CAN_main_group <- NA_character_
data0$CAN_main_group[as.character(data0$abuse) %in% main_can_types] <- "Main CAN types"
data0$CAN_main_group[as.character(data0$abuse) %in% other_reason_types] <- "Other consultation reasons"
data0$CAN_main_group <- factor(data0$CAN_main_group, levels = c("Other consultation reasons", "Main CAN types"))
data0$is_main_can <- ifelse(data0$CAN_main_group == "Main CAN types", 1, ifelse(data0$CAN_main_group == "Other consultation reasons", 0, NA))

data0$abuse_subtype4 <- ifelse(as.character(data0$abuse) %in% main_can_types, as.character(data0$abuse), NA)
data0$abuse_subtype4 <- factor(data0$abuse_subtype4, levels = main_can_types)

if ("age_year" %in% names(data0)) {
  data0$age_year <- safe_num(data0$age_year)
  data0$age_group <- cut(
    data0$age_year,
    breaks = c(-Inf, 6, 12, Inf),
    labels = c("0-6 years", "7-12 years", "13+ years"),
    right = TRUE
  )
}

if ("sex" %in% names(data0)) {
  data0$sex <- factor(data0$sex)
  data0$sex_male <- ifelse(data0$sex == "Male", 1, ifelse(data0$sex == "Female", 0, NA))
}


# -----------------------------
# 5. 論文の口腔関連変数に近い二値変数を作成
# -----------------------------
# このブロックでは、論文Table 2/3に対応する「う蝕あり」「永久歯充填あり」
# 「歯肉炎あり」「口腔清掃不良」などの有無変数を作る。
# これらは後続のカイ二乗検定とロジスティック回帰に用いる。

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
data0$Fair_or_poor_oral_hygiene <- ifelse(!is.na(data0$OralCleanStatus), as.integer(data0$OralCleanStatus %in% c("Poor", "Fair")), NA)
data0$Any_malocclusion <- ifelse(!is.na(data0$occlusalRelationship), as.integer(data0$occlusalRelationship != "Normal Occlusion"), NA)
data0$Oral_habit_present <- ifelse(!is.na(data0$habits), as.integer(data0$habits != "None"), NA)
data0$Dental_trauma_present <- ifelse(!is.na(data0$Trauma_total), as.integer(data0$Trauma_total > 0), NA)
data0$Retained_deciduous_present <- ifelse(!is.na(data0$Primary_retained), as.integer(data0$Primary_retained > 0), NA)
data0$Congenital_missing_present <- ifelse(!is.na(data0$Congenital_missing_total), as.integer(data0$Congenital_missing_total > 0), NA)
if ("Orthodontics" %in% names(data0)) {
  data0$Orthodontics_present <- ifelse(is.na(data0$Orthodontics), 0, as.integer(as.character(data0$Orthodontics) == "1"))
}

analysis_data_path <- file.path(OUT_DIR, "analysis_ready_data_with_derived_variables.csv")
write.csv(data0, analysis_data_path, row.names = FALSE)


# -----------------------------
# 6. 対象者フローと基本属性
# -----------------------------
# このブロックでは、解析対象数、比較群ごとの年齢・性別などを要約する。
# 論文Results冒頭の「平均年齢、性別、虐待理由の割合」に対応する。

flow <- data.frame(
  Step = c(
    "Loaded rows",
    "Rows with non-missing abuse",
    "Rows included in binary comparison",
    "Rows in main CAN subtype analysis"
  ),
  N = c(
    nrow(data0),
    sum(!is.na(data0$abuse)),
    sum(!is.na(data0$CAN_main_group)),
    sum(!is.na(data0$abuse_subtype4))
  ),
  stringsAsFactors = FALSE
)
write.csv(flow, file.path(OUT_DIR, "table0_analysis_flow.csv"), row.names = FALSE)

group_levels <- levels(data0$CAN_main_group)
demo_rows <- list()
demo_rows[[length(demo_rows) + 1]] <- data.frame(
  Variable = "N",
  Other_consultation_reasons = as.character(sum(data0$CAN_main_group == group_levels[1], na.rm = TRUE)),
  Main_CAN_types = as.character(sum(data0$CAN_main_group == group_levels[2], na.rm = TRUE)),
  Total = as.character(sum(!is.na(data0$CAN_main_group))),
  Test = "",
  p = "",
  stringsAsFactors = FALSE
)

if ("age_year" %in% names(data0)) {
  demo_rows[[length(demo_rows) + 1]] <- data.frame(
    Variable = "Age, years: mean +/- SD",
    Other_consultation_reasons = mean_sd(data0$age_year[data0$CAN_main_group == group_levels[1]]),
    Main_CAN_types = mean_sd(data0$age_year[data0$CAN_main_group == group_levels[2]]),
    Total = mean_sd(data0$age_year[!is.na(data0$CAN_main_group)]),
    Test = "Independent t-test",
    p = format_p(safe_t_p(data0$age_year, data0$CAN_main_group)),
    stringsAsFactors = FALSE
  )
  demo_rows[[length(demo_rows) + 1]] <- data.frame(
    Variable = "Age, years: median [IQR]",
    Other_consultation_reasons = median_iqr(data0$age_year[data0$CAN_main_group == group_levels[1]]),
    Main_CAN_types = median_iqr(data0$age_year[data0$CAN_main_group == group_levels[2]]),
    Total = median_iqr(data0$age_year[!is.na(data0$CAN_main_group)]),
    Test = "Wilcoxon sensitivity",
    p = format_p(safe_wilcox_p(data0$age_year, data0$CAN_main_group)),
    stringsAsFactors = FALSE
  )
}

if ("sex" %in% names(data0)) {
  sex_test <- safe_cat_test(data0$sex, data0$CAN_main_group)
  for (sx in levels(droplevels(data0$sex))) {
    n1 <- sum(data0$CAN_main_group == group_levels[1] & data0$sex == sx, na.rm = TRUE)
    d1 <- sum(data0$CAN_main_group == group_levels[1] & !is.na(data0$sex), na.rm = TRUE)
    n2 <- sum(data0$CAN_main_group == group_levels[2] & data0$sex == sx, na.rm = TRUE)
    d2 <- sum(data0$CAN_main_group == group_levels[2] & !is.na(data0$sex), na.rm = TRUE)
    nt <- sum(!is.na(data0$CAN_main_group) & data0$sex == sx, na.rm = TRUE)
    dt <- sum(!is.na(data0$CAN_main_group) & !is.na(data0$sex), na.rm = TRUE)
    demo_rows[[length(demo_rows) + 1]] <- data.frame(
      Variable = paste0("Sex: ", sx),
      Other_consultation_reasons = sprintf("%d/%d (%.1f%%)", n1, d1, ifelse(d1 > 0, n1 / d1 * 100, NA_real_)),
      Main_CAN_types = sprintf("%d/%d (%.1f%%)", n2, d2, ifelse(d2 > 0, n2 / d2 * 100, NA_real_)),
      Total = sprintf("%d/%d (%.1f%%)", nt, dt, ifelse(dt > 0, nt / dt * 100, NA_real_)),
      Test = sex_test$method,
      p = format_p(sex_test$p),
      stringsAsFactors = FALSE
    )
  }
}

if ("age_group" %in% names(data0)) {
  age_group_test <- safe_cat_test(data0$age_group, data0$CAN_main_group)
  for (ag in levels(droplevels(data0$age_group))) {
    n1 <- sum(data0$CAN_main_group == group_levels[1] & data0$age_group == ag, na.rm = TRUE)
    d1 <- sum(data0$CAN_main_group == group_levels[1] & !is.na(data0$age_group), na.rm = TRUE)
    n2 <- sum(data0$CAN_main_group == group_levels[2] & data0$age_group == ag, na.rm = TRUE)
    d2 <- sum(data0$CAN_main_group == group_levels[2] & !is.na(data0$age_group), na.rm = TRUE)
    nt <- sum(!is.na(data0$CAN_main_group) & data0$age_group == ag, na.rm = TRUE)
    dt <- sum(!is.na(data0$CAN_main_group) & !is.na(data0$age_group), na.rm = TRUE)
    demo_rows[[length(demo_rows) + 1]] <- data.frame(
      Variable = paste0("Age group: ", ag),
      Other_consultation_reasons = sprintf("%d/%d (%.1f%%)", n1, d1, ifelse(d1 > 0, n1 / d1 * 100, NA_real_)),
      Main_CAN_types = sprintf("%d/%d (%.1f%%)", n2, d2, ifelse(d2 > 0, n2 / d2 * 100, NA_real_)),
      Total = sprintf("%d/%d (%.1f%%)", nt, dt, ifelse(dt > 0, nt / dt * 100, NA_real_)),
      Test = age_group_test$method,
      p = format_p(age_group_test$p),
      stringsAsFactors = FALSE
    )
  }
}

table1_demographics <- do.call(rbind, demo_rows)
write.csv(table1_demographics, file.path(OUT_DIR, "table1_demographics_binary_groups.csv"), row.names = FALSE)


# -----------------------------
# 7. 連続口腔アウトカムの二群比較
# -----------------------------
# このブロックでは、論文Table 1に近い形で、乳歯・永久歯別の未処置う蝕、
# 喪失歯、処置歯、合計dmft/DMFTなどを平均±SDで示し、二群間をt検定で比較する。
# 歯数データは歪みやすいため、感度分析としてWilcoxon検定のp値も併記する。

continuous_vars <- data.frame(
  variable = c(
    "Primary_d", "Primary_m", "Primary_f", "Primary_dmft",
    "Perm_D", "Perm_M", "Perm_F", "Perm_DMFT",
    "Decayed_total", "Missing_total", "Filled_total", "Total_dmft_DMFT",
    "C0_total", "Trauma_total", "Primary_retained", "Congenital_missing_total",
    "Healthy_rate", "Care_index", "Untreated_caries_rate"
  ),
  label = c(
    "Primary dentition: decayed", "Primary dentition: missing", "Primary dentition: filled", "Primary dentition: dmft total",
    "Permanent dentition: decayed", "Permanent dentition: missing", "Permanent dentition: filled", "Permanent dentition: DMFT total",
    "Total decayed teeth", "Total missing teeth", "Total filled teeth", "Total dmft/DMFT",
    "C0 count", "Dental trauma count", "Retained deciduous teeth", "Congenital missing teeth",
    "Healthy teeth rate (%)", "Care index (%) among dmft/DMFT>0", "Untreated caries rate (%) among dmft/DMFT>0"
  ),
  stringsAsFactors = FALSE
)
continuous_vars <- continuous_vars[continuous_vars$variable %in% names(data0), , drop = FALSE]

cont_rows <- list()
for (i in seq_len(nrow(continuous_vars))) {
  v <- continuous_vars$variable[i]
  x <- data0[[v]]
  g <- data0$CAN_main_group
  if (v %in% c("Care_index", "Untreated_caries_rate")) {
    valid_dmft <- !is.na(data0$Total_dmft_DMFT) & data0$Total_dmft_DMFT > 0
  } else {
    valid_dmft <- rep(TRUE, nrow(data0))
  }
  cont_rows[[length(cont_rows) + 1]] <- data.frame(
    Variable = continuous_vars$label[i],
    Other_N = sum(g == group_levels[1] & !is.na(x) & valid_dmft, na.rm = TRUE),
    Other_Mean_SD = mean_sd(x[g == group_levels[1] & valid_dmft]),
    Other_Median_IQR = median_iqr(x[g == group_levels[1] & valid_dmft]),
    Main_CAN_N = sum(g == group_levels[2] & !is.na(x) & valid_dmft, na.rm = TRUE),
    Main_CAN_Mean_SD = mean_sd(x[g == group_levels[2] & valid_dmft]),
    Main_CAN_Median_IQR = median_iqr(x[g == group_levels[2] & valid_dmft]),
    Total_N = sum(!is.na(g) & !is.na(x) & valid_dmft, na.rm = TRUE),
    Total_Mean_SD = mean_sd(x[!is.na(g) & valid_dmft]),
    Total_Median_IQR = median_iqr(x[!is.na(g) & valid_dmft]),
    Test_primary = "Independent t-test",
    p_t_test = format_p(safe_t_p(x[valid_dmft], g[valid_dmft])),
    p_wilcoxon_sensitivity = format_p(safe_wilcox_p(x[valid_dmft], g[valid_dmft])),
    stringsAsFactors = FALSE
  )
}
table2_continuous <- do.call(rbind, cont_rows)
write.csv(table2_continuous, file.path(OUT_DIR, "table2_continuous_oral_health_binary_groups.csv"), row.names = FALSE)


# -----------------------------
# 8. カテゴリ口腔アウトカムの二群比較
# -----------------------------
# このブロックでは、論文Table 2に近い形で、う蝕あり、治療必要、
# 歯肉炎、口腔清掃不良などの割合を示し、カイ二乗検定またはFisher正確検定で比較する。

categorical_vars <- c(
  "Primary_caries_present",
  "Permanent_caries_present",
  "Any_caries_experience",
  "Untreated_caries_present",
  "Permanent_fillings_present",
  "Primary_fillings_present",
  "Treatment_need",
  "Urgent_treatment",
  "Gingivitis_present",
  "Poor_oral_hygiene",
  "Fair_or_poor_oral_hygiene",
  "Any_malocclusion",
  "Oral_habit_present",
  "Dental_trauma_present",
  "Retained_deciduous_present",
  "Congenital_missing_present",
  "Orthodontics_present"
)
categorical_vars <- categorical_vars[categorical_vars %in% names(data0)]
categorical_labels <- c(
  Primary_caries_present = "Dental caries in primary teeth",
  Permanent_caries_present = "Dental caries in permanent teeth",
  Any_caries_experience = "Any caries experience (dmft/DMFT>0)",
  Untreated_caries_present = "Untreated caries present",
  Permanent_fillings_present = "Filled permanent teeth present",
  Primary_fillings_present = "Filled primary teeth present",
  Treatment_need = "Treatment required",
  Urgent_treatment = "Urgent treatment required",
  Gingivitis_present = "Gingivitis",
  Poor_oral_hygiene = "Poor oral hygiene",
  Fair_or_poor_oral_hygiene = "Fair or poor oral hygiene",
  Any_malocclusion = "Any malocclusion",
  Oral_habit_present = "Oral habit present",
  Dental_trauma_present = "Dental trauma",
  Retained_deciduous_present = "Retained deciduous teeth",
  Congenital_missing_present = "Congenital missing teeth",
  Orthodontics_present = "Orthodontic finding recorded"
)

cat_rows <- list()
for (v in categorical_vars) {
  x <- data0[[v]]
  g <- data0$CAN_main_group
  test <- safe_cat_test(x, g)
  n1 <- sum(g == group_levels[1] & x == 1, na.rm = TRUE)
  d1 <- sum(g == group_levels[1] & !is.na(x), na.rm = TRUE)
  n2 <- sum(g == group_levels[2] & x == 1, na.rm = TRUE)
  d2 <- sum(g == group_levels[2] & !is.na(x), na.rm = TRUE)
  nt <- sum(!is.na(g) & x == 1, na.rm = TRUE)
  dt <- sum(!is.na(g) & !is.na(x), na.rm = TRUE)
  cat_rows[[length(cat_rows) + 1]] <- data.frame(
    Variable = ifelse(v %in% names(categorical_labels), categorical_labels[[v]], v),
    Other_consultation_reasons = sprintf("%d/%d (%.1f%%)", n1, d1, ifelse(d1 > 0, n1 / d1 * 100, NA_real_)),
    Main_CAN_types = sprintf("%d/%d (%.1f%%)", n2, d2, ifelse(d2 > 0, n2 / d2 * 100, NA_real_)),
    Total = sprintf("%d/%d (%.1f%%)", nt, dt, ifelse(dt > 0, nt / dt * 100, NA_real_)),
    Test = test$method,
    p = format_p(test$p),
    p_value = test$p,
    stringsAsFactors = FALSE
  )
}
table3_categorical <- do.call(rbind, cat_rows)
write.csv(table3_categorical, file.path(OUT_DIR, "table3_categorical_oral_health_binary_groups.csv"), row.names = FALSE)


# -----------------------------
# 9. 主要CAN 4分類内のサブタイプ比較
# -----------------------------
# このブロックでは、論文本文の「CANサブタイプ間で口腔変数に有意差があるか」
# という確認に対応し、4つの虐待種別間で連続変数はANOVA/Kruskal-Wallis、
# カテゴリ変数はカイ二乗/Fisher検定を行う。

subtype_data <- data0[!is.na(data0$abuse_subtype4), , drop = FALSE]

subtype_cont_rows <- list()
for (i in seq_len(nrow(continuous_vars))) {
  v <- continuous_vars$variable[i]
  if (!(v %in% names(subtype_data))) next
  x <- subtype_data[[v]]
  g <- subtype_data$abuse_subtype4
  row <- data.frame(Variable = continuous_vars$label[i], stringsAsFactors = FALSE)
  for (ab in main_can_types) {
    row[[paste0(ab, "_N")]] <- sum(g == ab & !is.na(x), na.rm = TRUE)
    row[[paste0(ab, "_Mean_SD")]] <- mean_sd(x[g == ab])
    row[[paste0(ab, "_Median_IQR")]] <- median_iqr(x[g == ab])
  }
  row$Test_primary <- "One-way ANOVA"
  row$p_anova <- format_p(safe_anova_p(x, g))
  row$p_kruskal_sensitivity <- format_p(safe_kruskal_p(x, g))
  subtype_cont_rows[[length(subtype_cont_rows) + 1]] <- row
}
table4_subtype_continuous <- if (length(subtype_cont_rows) > 0) do.call(rbind, subtype_cont_rows) else data.frame()
write.csv(table4_subtype_continuous, file.path(OUT_DIR, "table4_subtype_continuous_tests.csv"), row.names = FALSE)

subtype_cat_rows <- list()
for (v in categorical_vars) {
  if (!(v %in% names(subtype_data))) next
  x <- subtype_data[[v]]
  g <- subtype_data$abuse_subtype4
  test <- safe_cat_test(x, g)
  row <- data.frame(
    Variable = ifelse(v %in% names(categorical_labels), categorical_labels[[v]], v),
    Test = test$method,
    p = format_p(test$p),
    p_value = test$p,
    stringsAsFactors = FALSE
  )
  for (ab in main_can_types) {
    n <- sum(g == ab & x == 1, na.rm = TRUE)
    d <- sum(g == ab & !is.na(x), na.rm = TRUE)
    row[[ab]] <- sprintf("%d/%d (%.1f%%)", n, d, ifelse(d > 0, n / d * 100, NA_real_))
  }
  subtype_cat_rows[[length(subtype_cat_rows) + 1]] <- row
}
table5_subtype_categorical <- if (length(subtype_cat_rows) > 0) do.call(rbind, subtype_cat_rows) else data.frame()
write.csv(table5_subtype_categorical, file.path(OUT_DIR, "table5_subtype_categorical_tests.csv"), row.names = FALSE)


# -----------------------------
# 10. ロジスティック回帰: 単変量
# -----------------------------
# このブロックでは、論文と同様に「調査対象群であること」を従属変数とする
# 単変量ロジスティック回帰を行う。本データでは従属変数を
# 主要CAN 4分類(1) vs その他相談理由(0)とする。

logistic_candidates <- c(
  "Primary_caries_present",
  "Permanent_caries_present",
  "Permanent_fillings_present",
  "Primary_fillings_present",
  "Treatment_need",
  "Urgent_treatment",
  "Gingivitis_present",
  "Poor_oral_hygiene",
  "Fair_or_poor_oral_hygiene",
  "Any_malocclusion",
  "Oral_habit_present",
  "Dental_trauma_present",
  "Retained_deciduous_present",
  "Congenital_missing_present",
  "Orthodontics_present"
)
logistic_candidates <- logistic_candidates[logistic_candidates %in% names(data0)]
logistic_labels <- categorical_labels[logistic_candidates]
names(logistic_labels) <- logistic_candidates

univ_rows <- list()
for (v in logistic_candidates) {
  model_df <- data0[, c("is_main_can", v), drop = FALSE]
  model_df <- model_df[complete.cases(model_df), , drop = FALSE]
  if (nrow(model_df) < 30) next
  if (length(unique(model_df$is_main_can)) < 2 || length(unique(model_df[[v]])) < 2) next
  form <- as.formula(paste("is_main_can ~", v))
  fit <- try(glm(form, data = model_df, family = binomial()), silent = TRUE)
  if (inherits(fit, "try-error")) next
  tab <- logistic_or_table(fit, paste0("Univariate: ", logistic_labels[[v]]), logistic_labels)
  if (nrow(tab) > 0) {
    tab$Predictor <- v
    tab$N <- nrow(model_df)
    tab$Events_main_CAN <- sum(model_df$is_main_can == 1)
    univ_rows[[length(univ_rows) + 1]] <- tab
  }
}
table6_univariate_logistic <- if (length(univ_rows) > 0) do.call(rbind, univ_rows) else data.frame()
write.csv(table6_univariate_logistic, file.path(OUT_DIR, "table6_univariate_logistic.csv"), row.names = FALSE)


# -----------------------------
# 11. ロジスティック回帰: 多変量
# -----------------------------
# このブロックでは、論文と同じ考え方で、単変量解析でp<0.05だった口腔関連因子を
# 多変量ロジスティック回帰に投入する。年齢と性別は、このデータではマッチング対照が
# 無いため、利用可能な場合は調整因子として加える。

selected_predictors <- character(0)
if (nrow(table6_univariate_logistic) > 0) {
  selected_predictors <- unique(table6_univariate_logistic$Predictor[table6_univariate_logistic$p_value < 0.05])
}

adjust_terms <- character(0)
if ("age_year" %in% names(data0)) adjust_terms <- c(adjust_terms, "age_year")
if ("sex_male" %in% names(data0)) adjust_terms <- c(adjust_terms, "sex_male")

table7_multivariable_logistic <- data.frame()
fit_multi <- NULL
if (length(selected_predictors) > 0) {
  model_terms <- c(adjust_terms, selected_predictors)
  model_df <- data0[, c("is_main_can", model_terms), drop = FALSE]
  model_df <- model_df[complete.cases(model_df), , drop = FALSE]
  variable_has_variation <- vapply(model_terms, function(v) length(unique(model_df[[v]])) >= 2, logical(1))
  model_terms <- model_terms[variable_has_variation]
  selected_predictors <- selected_predictors[selected_predictors %in% model_terms]
  if (nrow(model_df) >= 50 && length(unique(model_df$is_main_can)) == 2 && length(selected_predictors) > 0) {
    form <- as.formula(paste("is_main_can ~", paste(model_terms, collapse = " + ")))
    fit_multi <- try(glm(form, data = model_df[, c("is_main_can", model_terms), drop = FALSE], family = binomial()), silent = TRUE)
    if (!inherits(fit_multi, "try-error")) {
      lookup <- c(logistic_labels, age_year = "Age, years", sex_male = "Male sex")
      table7_multivariable_logistic <- logistic_or_table(fit_multi, "Multivariable logistic", lookup)
      table7_multivariable_logistic$N <- nrow(model_df)
      table7_multivariable_logistic$Events_main_CAN <- sum(model_df$is_main_can == 1)
      table7_multivariable_logistic$Selected_from_univariate <- table7_multivariable_logistic$Term %in% selected_predictors
    }
  }
}
write.csv(table7_multivariable_logistic, file.path(OUT_DIR, "table7_multivariable_logistic.csv"), row.names = FALSE)


# -----------------------------
# 12. 累積予測確率
# -----------------------------
# このブロックでは、論文Fig. 1の考え方をまねて、多変量モデルで残った有意な
# 口腔関連因子が0個、1個、2個...と存在するときの予測確率を計算する。
# ここでの確率は「主要CAN 4分類に属する確率」であり、一般対照との比較ではない。

probability_table <- data.frame()
if (!is.null(fit_multi) && !inherits(fit_multi, "try-error") && nrow(table7_multivariable_logistic) > 0) {
  significant_terms <- table7_multivariable_logistic$Term[
    table7_multivariable_logistic$Selected_from_univariate &
      table7_multivariable_logistic$p_value < 0.05 &
      table7_multivariable_logistic$Term %in% selected_predictors
  ]
  if (length(significant_terms) > 0) {
    significant_terms <- significant_terms[order(
      table7_multivariable_logistic$OR[match(significant_terms, table7_multivariable_logistic$Term)],
      decreasing = TRUE
    )]
    model_terms <- all.vars(formula(fit_multi))[-1]
    base_row <- as.data.frame(as.list(rep(0, length(model_terms))), stringsAsFactors = FALSE)
    names(base_row) <- model_terms
    if ("age_year" %in% names(base_row)) base_row$age_year <- mean(data0$age_year, na.rm = TRUE)
    if ("sex_male" %in% names(base_row)) {
      base_row$sex_male <- as.numeric(names(which.max(table(data0$sex_male, useNA = "no"))))
    }
    for (v in selected_predictors) {
      if (v %in% names(base_row)) base_row[[v]] <- 0
    }
    scenarios <- list()
    scenarios[[length(scenarios) + 1]] <- base_row
    scenario_labels <- "No significant oral-health factor present"
    running_row <- base_row
    for (v in significant_terms) {
      running_row[[v]] <- 1
      scenarios[[length(scenarios) + 1]] <- running_row
      scenario_labels <- c(scenario_labels, paste("Plus", paste(significant_terms[seq_len(match(v, significant_terms))], collapse = " + ")))
    }
    for (i in seq_along(scenarios)) {
      pred <- predict(fit_multi, newdata = scenarios[[i]], type = "response")
      probability_table <- rbind(
        probability_table,
        data.frame(
          Scenario = scenario_labels[i],
          Probability_main_CAN = round(as.numeric(pred), 4),
          Probability_percent = round(as.numeric(pred) * 100, 1),
          stringsAsFactors = FALSE
        )
      )
    }
  }
}
if (nrow(probability_table) == 0) {
  probability_table <- data.frame(
    Note = "No oral-health predictor selected from univariate analysis remained statistically significant in the multivariable model; cumulative predicted probability was not calculated.",
    stringsAsFactors = FALSE
  )
}
write.csv(probability_table, file.path(OUT_DIR, "table8_cumulative_predicted_probability.csv"), row.names = FALSE)


# -----------------------------
# 13. 図の作成
# -----------------------------
# このブロックでは、主要なアウトカムの分布を確認するため、
# 合計dmft/DMFTの箱ひげ図と、主要二値アウトカムの割合図を作成する。

if ("Total_dmft_DMFT" %in% names(data0)) {
  png(file.path(OUT_DIR, "figure1_total_dmft_DMFT_by_binary_group.png"), width = 1800, height = 1200, res = 200)
  boxplot(
    Total_dmft_DMFT ~ CAN_main_group,
    data = data0,
    main = "Total dmft/DMFT by analysis group",
    xlab = "",
    ylab = "Total dmft/DMFT",
    col = c("#d8e2dc", "#ffd6a5")
  )
  dev.off()

  if (nrow(subtype_data) > 0) {
    png(file.path(OUT_DIR, "figure2_total_dmft_DMFT_by_abuse_subtype.png"), width = 2200, height = 1200, res = 200)
    boxplot(
      Total_dmft_DMFT ~ abuse_subtype4,
      data = subtype_data,
      main = "Total dmft/DMFT by main CAN subtype",
      xlab = "",
      ylab = "Total dmft/DMFT",
      col = c("#b7e4c7", "#a9def9", "#f6bd60", "#f28482")
    )
    dev.off()
  }
}

plot_vars <- c("Primary_caries_present", "Permanent_fillings_present", "Treatment_need", "Gingivitis_present", "Poor_oral_hygiene")
plot_vars <- plot_vars[plot_vars %in% names(data0)]
if (length(plot_vars) > 0) {
  prevalence <- matrix(NA_real_, nrow = length(plot_vars), ncol = length(group_levels))
  rownames(prevalence) <- unname(categorical_labels[plot_vars])
  colnames(prevalence) <- group_levels
  for (i in seq_along(plot_vars)) {
    for (j in seq_along(group_levels)) {
      denom <- sum(data0$CAN_main_group == group_levels[j] & !is.na(data0[[plot_vars[i]]]), na.rm = TRUE)
      num <- sum(data0$CAN_main_group == group_levels[j] & data0[[plot_vars[i]]] == 1, na.rm = TRUE)
      prevalence[i, j] <- ifelse(denom > 0, num / denom * 100, NA_real_)
    }
  }
  png(file.path(OUT_DIR, "figure3_key_binary_outcome_prevalence.png"), width = 2200, height = 1400, res = 200)
  par(mar = c(9, 5, 3, 1))
  barplot(
    t(prevalence),
    beside = TRUE,
    las = 2,
    ylim = c(0, max(prevalence, na.rm = TRUE) * 1.2),
    ylab = "Prevalence (%)",
    col = c("#d8e2dc", "#ffd6a5"),
    legend.text = TRUE,
    args.legend = list(x = "topright", bty = "n")
  )
  title("Key oral-health variables by analysis group")
  dev.off()
}


# -----------------------------
# 14. 解析メモを書き出し
# -----------------------------
# このブロックでは、論文から真似した統計手順と、このデータに合わせた変更点を
# テキストとして保存する。

notes <- c(
  "Paper-like analysis notes",
  "=========================",
  "",
  "Reference method mimicked from Kvist et al. (2018):",
  "- Retrospective dental-record based analysis.",
  "- Continuous variables are summarized as mean +/- SD and compared by independent t-test.",
  "- Nominal variables are compared by chi-square test; Fisher exact test is used when expected counts are small.",
  "- Univariate and multivariable logistic regression estimate associations between oral-health variables and group status.",
  "- Predictors significant in multivariable logistic regression are used to estimate cumulative predicted probability.",
  "",
  "Adaptation for this dataset:",
  "- No matched healthy control group is available in analysisData_20260211_AllData_cleaned.csv.",
  "- The binary outcome is therefore Main CAN types (Physical Abuse, Neglect, Emotional Abuse, Sexual Abuse) vs Other consultation reasons (Delinquency, Parenting Difficulties, Others).",
  "- Abuse subtype analysis compares the four main CAN types.",
  "- C0 is treated as incipient caries and not included in dmft/DMFT, matching the paper's definition of caries requiring treatment.",
  "- Tooth code 4 is counted as missing/m, following the supplied tooth coding; whether all missing teeth are caries-related should be confirmed from the data dictionary."
)
writeLines(notes, file.path(OUT_DIR, "analysis_notes.txt"))

message("Analysis complete. Output directory: ", OUT_DIR)
