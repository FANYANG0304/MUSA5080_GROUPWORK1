# =========================================================
# 一次性完整脚本：偏度检查 → 对数变换 → 相关性矩阵 → VIF
# → Lasso(保留 y_coord) → 线性模型 + 10 折 CV + 预测图
# =========================================================
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(e1071)
library(purrr)
library(patchwork)
library(corrplot)
library(reshape2)
library(car)
library(glmnet)
library(here)
library(caret)

set.seed(2025)
# 如需固定项目根目录，取消下一行注释：
# setwd("Users/zhaozhiyuan/Documents/GitHub/MUSA5080_GROUPWORK1")

# ========== 公用函数 ==========
safe_log <- function(x) {
  x <- as.numeric(x)
  if (all(is.na(x))) return(x)
  if (all(is.na(x) | is.infinite(x))) return(x)
  if (min(x, na.rm = TRUE) >= 0) return(log1p(x))
  shift <- -min(x, na.rm = TRUE) + 1
  log(x + shift)
}

# 构造公式（只保留数据中存在的变量；交互项需两边变量都在）
make_formula <- function(y, main_terms = character(), inter_terms = list(), fe_terms = character(), data) {
  terms <- intersect(main_terms, names(data))
  # 交互项
  inter_ok <- c()
  for (pr in inter_terms) {
    if (length(pr) == 2 && all(pr %in% names(data))) {
      inter_ok <- c(inter_ok, paste0(pr[1], ":", pr[2]))
    }
  }
  # 固定效应（因子列）
  fe_ok <- intersect(fe_terms, names(data))
  rhs_parts <- c(terms, inter_ok, fe_ok)
  rhs <- if (length(rhs_parts)) paste(rhs_parts, collapse = " + ") else "1"
  as.formula(paste(y, "~", rhs))
}

# =========================================================
# Step 1: 读取数据与初步处理（新增 age）
# =========================================================
df <- read_csv( "data/opa_sales_final_complete.csv", show_col_types = FALSE)

# 房龄
if ("year_built" %in% names(df)) {
  df <- df %>% mutate(age = 2025 - year_built) %>% select(-year_built)
}

# 类型清理：zip_code 若存在，设为因子
if ("zip_code" %in% names(df) && !is.factor(df$zip_code)) {
  if (is.numeric(df$zip_code)) df$zip_code <- as.character(df$zip_code)
  df$zip_code <- as.factor(df$zip_code)
}

# 数值列快照用于偏度图
num0 <- df %>% select(where(is.numeric))

# =========================================================
# Step 2: 偏度检查与对数变换前后对比图（仅可视化）
# =========================================================
dir.create("plots_before_after", showWarnings = FALSE)

skew_tbl <- num0 %>%
  summarise(across(everything(), ~ skewness(.x, na.rm = TRUE))) %>%
  pivot_longer(everything(), names_to = "variable", values_to = "skew_before") %>%
  mutate(needs_log = abs(skew_before) >= 1)

vars_high <- skew_tbl %>% filter(needs_log) %>% pull(variable)
skew_after_list <- list()

for (v in vars_high) {
  x <- num0[[v]]
  x_log <- safe_log(x)
  sk_b <- skewness(x, na.rm = TRUE)
  sk_a <- skewness(x_log, na.rm = TRUE)
  skew_after_list[[v]] <- sk_a
  
  p1 <- ggplot(data.frame(x = x), aes(x = x)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "#9ecae1", color = "white") +
    geom_density(size = 1) +
    labs(title = paste0(v, " | Before (skew=", round(sk_b, 2), ")"),
         x = v, y = "Density") + theme_minimal(base_size = 12)
  
  p2 <- ggplot(data.frame(x = x_log), aes(x = x)) +
    geom_histogram(aes(y = ..density..), bins = 30, fill = "#fc9272", color = "white") +
    geom_density(size = 1) +
    labs(title = paste0(v, " | After log (skew=", round(sk_a, 2), ")"),
         x = paste0("log-transformed ", v), y = "Density") + theme_minimal(base_size = 12)
  
  ggsave(file.path("plots_before_after", paste0("compare_", v, ".png")),
         p1 + p2 + plot_layout(ncol = 2), width = 12, height = 4.5, dpi = 150)
}

skew_after_tbl <- tibble(variable = names(skew_after_list),
                         skew_after = unlist(skew_after_list, use.names = FALSE))
skew_compare <- skew_tbl %>%
  left_join(skew_after_tbl, by = "variable") %>%
  mutate(improved = ifelse(!is.na(skew_after) & abs(skew_after) < abs(skew_before), "Yes", "No"))
write.csv(skew_compare, "skewness_before_after_summary.csv", row.names = FALSE)

# =========================================================
# Step 3: 先做 log，再删原始列与 category_code（确保 sale_price_log 存在）
# =========================================================
cols_to_log <- c("sale_price",
                 "dist_to_park_ft",
                 "dist_to_hospital_ft",
                 "dist_transit_ft",
                 "per_cap_incomeE",
                 "dist_to_nearest_school_ft",
                 "total_livable_area")

# 先生成 *_log
df <- df %>%
  mutate(across(all_of(intersect(cols_to_log, names(.))),
                ~ safe_log(.x), .names = "{.col}_log"))

# 再删除原始列与 category_code
df <- df %>% select(-any_of(c(cols_to_log, "category_code")))

# 写出供复用
if (!dir.exists("data")) dir.create("data", recursive = TRUE)
write_csv(df, "data/opa_sales_with_logs.csv")
message("完成 log 变换并删除原始列 → data/opa_sales_with_logs.csv")

# =========================================================
# Step 4: 相关矩阵与热图
# =========================================================
num_df <- df %>% select(where(is.numeric))
cor_mat <- cor(num_df, use = "pairwise.complete.obs")
write.csv(cor_mat, "data/correlation_matrix.csv", row.names = TRUE)

corrplot(cor_mat, method = "color", type = "upper",
         tl.col = "black", tl.cex = 0.6,
         col = colorRampPalette(c("blue","white","red"))(200),
         title = "Correlation Matrix", mar = c(0,0,1,0))

cor_melt <- melt(cor_mat)
p_cor <- ggplot(cor_melt, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
  theme_minimal(base_size = 9) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(x = NULL, y = NULL, fill = "Corr")
ggsave("data/correlation_heatmap.png", p_cor, width = 10, height = 8, dpi = 150)

# =========================================================
# Step 5: 多重共线性 (VIF)
# =========================================================
num <- df %>% select(where(is.numeric))
stopifnot("sale_price_log" %in% names(num))
y <- "sale_price_log"
X <- setdiff(names(num), y)

# 零方差剔除
nzv <- sapply(num[X], function(z) sd(z, na.rm = TRUE) == 0)
X <- setdiff(X, names(nzv)[nzv])

# 相关性预筛
cm <- cor(num[X], use = "pairwise.complete.obs")
high_pairs <- which(abs(cm) > 0.9 & upper.tri(cm), arr.ind = TRUE)
if (nrow(high_pairs) > 0) {
  rm_vars <- unique(colnames(cm)[high_pairs[,2]])
  X <- setdiff(X, rm_vars)
}

mk_formula <- function(features) as.formula(paste(y, "~", paste(features, collapse = " + ")))
iter_log <- list(); features <- X; iter <- 0
repeat {
  iter <- iter + 1
  fit <- lm(mk_formula(features), data = num)
  vt <- data.frame(variable = names(car::vif(fit)),
                   VIF = as.numeric(car::vif(fit)), row.names = NULL) |>
    arrange(desc(VIF))
  vt$iter <- iter
  iter_log[[iter]] <- vt
  if (max(vt$VIF, na.rm = TRUE) <= 5 || length(features) <= 1) break
  features <- setdiff(features, vt$variable[which.max(vt$VIF)])
}
vif_history <- dplyr::bind_rows(iter_log) |> select(iter, variable, VIF)
write.csv(vif_history, "data/vif_history.csv", row.names = FALSE)

final_fit <- lm(mk_formula(features), data = num)
final_vif <- data.frame(variable = names(car::vif(final_fit)),
                        VIF = as.numeric(car::vif(final_fit)), row.names = NULL) |>
  arrange(desc(VIF))
write.csv(final_vif, "data/vif_final.csv", row.names = FALSE)
cat("Final formula:\n", deparse(mk_formula(features)), "\n")
print(final_vif)

# =========================================================
# Step 6: Lasso（保留 y_coord 不受惩罚）
# =========================================================
x <- as.matrix(num[, features, drop = FALSE])
y_vec <- num[[y]]
ok <- complete.cases(x, y_vec)
x <- x[ok, , drop = FALSE]; y_vec <- y_vec[ok]

if (!"y_coord" %in% colnames(x)) stop("y_coord 不在特征集中。")
pf <- rep(1, ncol(x)); pf[colnames(x) == "y_coord"] <- 0

cvfit_keepY <- cv.glmnet(
  x = x, y = y_vec,
  alpha = 1, family = "gaussian",
  nfolds = 10, standardize = TRUE,
  penalty.factor = pf
)
cat("lambda.min =", cvfit_keepY$lambda.min, "\n")
cat("lambda.1se =", cvfit_keepY$lambda.1se, "\n\n")

coef_1se <- as.matrix(coef(cvfit_keepY, s = "lambda.1se"))
nz_keepY <- data.frame(term = rownames(coef_1se),
                       coef = coef_1se[,1], row.names = NULL) |>
  dplyr::filter(coef != 0, term != "(Intercept)")
cat("=== Lasso (λ_1se) + 保留 y_coord ===\n")
print(nz_keepY)

# =========================================================
# Step 7: 线性模型构建 + 10 折交叉验证 + 预测图
# =========================================================
data_clean <- df  # 直接复用已处理数据

# 备选变量组
struct_vars <- c("number_of_bathrooms", "total_livable_area_log",
                 "exterior_condition", "interior_condition")
census_vars <- c("median_incomeE", "PCTPOVERTY")
spatial_vars <- c("x_coord", "y_coord", "hospitals_15min_walk")
fe_vars     <- c("zip_code")

# 交互对
inter_pairs <- list(c("number_of_bathrooms", "total_livable_area_log"),
                    c("median_incomeE", "PCTPOVERTY"))

# 动态公式
f1 <- make_formula("sale_price_log", main_terms = struct_vars, data = data_clean)
f2 <- make_formula("sale_price_log", main_terms = c(struct_vars, census_vars), data = data_clean)
f3 <- make_formula("sale_price_log", main_terms = c(struct_vars, census_vars, spatial_vars), data = data_clean)
f4 <- make_formula("sale_price_log", main_terms = c(struct_vars, census_vars, spatial_vars),
                   inter_terms = inter_pairs, fe_terms = fe_vars, data = data_clean)

# 拟合
m1 <- lm(f1, data = data_clean)
m2 <- lm(f2, data = data_clean)
m3 <- lm(f3, data = data_clean)
m4 <- lm(f4, data = data_clean)

# 10 折 CV（基于公式）
cv_ctrl <- trainControl(method = "cv", number = 10)
cv1 <- train(f1, data = data_clean, method = "lm", trControl = cv_ctrl, na.action = na.omit)
cv2 <- train(f2, data = data_clean, method = "lm", trControl = cv_ctrl, na.action = na.omit)
cv3 <- train(f3, data = data_clean, method = "lm", trControl = cv_ctrl, na.action = na.omit)
cv4 <- train(f4, data = data_clean, method = "lm", trControl = cv_ctrl, na.action = na.omit)

cv_summary <- data.frame(
  Model = c("Structural Only", "+ Census", "+ Spatial", "+ Interactions/FE"),
  CV_RMSE   = c(cv1$results$RMSE[1], cv2$results$RMSE[1], cv3$results$RMSE[1], cv4$results$RMSE[1]),
  R_squared = c(cv1$results$Rsquared[1], cv2$results$Rsquared[1], cv3$results$Rsquared[1], cv4$results$Rsquared[1])
)
write.csv(cv_summary, "model_performance_summary.csv", row.names = FALSE)
print(cv_summary)

# 预测图
pred_df <- list(
  "Structural Only"  = predict(m1, data_clean),
  "+ Census"         = predict(m2, data_clean),
  "+ Spatial"        = predict(m3, data_clean),
  "+ Interactions/FE"= predict(m4, data_clean)
)
plot_data <- bind_rows(lapply(names(pred_df), function(nm) {
  tibble(actual = data_clean$sale_price_log, predicted = pred_df[[nm]], model = nm)
}))
p_pred <- ggplot(plot_data, aes(actual, predicted)) +
  geom_point(alpha = 0.3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  facet_wrap(~model, ncol = 2) +
  labs(x = "Actual (log)", y = "Predicted (log)", title = "Predicted vs Actual") +
  theme_minimal()
ggsave("prediction_vs_actual.png", p_pred, width = 12, height = 8, dpi = 300)

# 摘要
cat("\n=== OLS Summaries ===\n")
print(summary(m1)); print(summary(m2)); print(summary(m3)); print(summary(m4))
cat("\n文件输出：\n",
    "- data/opa_sales_with_logs.csv\n",
    "- data/correlation_matrix.csv, data/correlation_heatmap.png\n",
    "- data/vif_history.csv, data/vif_final.csv\n",
    "- skewness_before_after_summary.csv\n",
    "- model_performance_summary.csv, prediction_vs_actual.png\n")

# =========================================================
# Phase 6: Model Diagnostics (Technical Appendix Only)
# 选择 CV RMSE 最小的模型进行诊断，输出三张图和解释建议
# 依赖：lmtest、sandwich
# =========================================================
library(lmtest)
library(sandwich)

# 1) 选取最佳模型（以 CV_RMSE 最小为准）
models_list <- list(m1 = m1, m2 = m2, m3 = m3, m4 = m4)
best_idx <- which.min(cv_summary$CV_RMSE)
best_key <- c("m1","m2","m3","m4")[best_idx]
best_model <- models_list[[best_key]]
cat("Best model by CV RMSE:", best_key, "\n")

# 2) 生成诊断数据
diag_dir <- "diagnostics"
if (!dir.exists(diag_dir)) dir.create(diag_dir, recursive = TRUE)

fit   <- fitted(best_model)
res   <- residuals(best_model)
rstd  <- rstandard(best_model)
cooks <- cooks.distance(best_model)
n_obs <- length(res)
cook_thr <- 4 / n_obs

df_res <- data.frame(fitted = fit, resid = res)
df_qq  <- data.frame(sample = rstd)
df_ck  <- data.frame(idx = seq_len(n_obs), cook = cooks, flag = cooks > cook_thr)

# 3) 残差图：线性与同方差性
p_res <- ggplot(df_res, aes(fitted, resid)) +
  geom_point(alpha = 0.35) +
  geom_smooth(method = "loess", se = FALSE) +
  geom_hline(yintercept = 0) +
  labs(x = "Fitted values", y = "Residuals",
       title = paste0("Residuals vs Fitted (", best_key, ")")) +
  theme_minimal()
ggsave(file.path(diag_dir, "residuals_vs_fitted.png"), p_res, width = 7, height = 5, dpi = 150)

# 4) Q–Q 图：正态性
p_qq <- ggplot(df_qq, aes(sample = sample)) +
  stat_qq() + stat_qq_line() +
  labs(title = paste0("Normal Q–Q Plot (", best_key, ")"),
       x = "Theoretical Quantiles", y = "Standardized Residuals") +
  theme_minimal()
ggsave(file.path(diag_dir, "qq_plot.png"), p_qq, width = 7, height = 5, dpi = 150)

# 5) Cook's distance：影响点
p_cook <- ggplot(df_ck, aes(idx, cook)) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = cook_thr, linetype = "dashed") +
  labs(title = paste0("Cook's Distance (", best_key, ")"),
       x = "Observation index", y = "Cook's D") +
  theme_minimal()
ggsave(file.path(diag_dir, "cooks_distance.png"), p_cook, width = 7, height = 5, dpi = 150)

# 6) 统计检验与解释提示
bp <- bptest(best_model)  # 同方差性检验
sw_p <- tryCatch(shapiro.test(res)$p.value, error = function(e) NA_real_)  # 正态性（n大时仅作参考）

cat("\n=== Diagnostics Summary ===\n")
cat("Breusch-Pagan p-value (homoscedasticity):", signif(bp$p.value, 4), "\n")
if (!is.na(sw_p)) cat("Shapiro-Wilk p-value (normality, heuristic):", signif(sw_p, 4), "\n")

n_influential <- sum(df_ck$flag)
cat("Influential points (Cook's D >", round(cook_thr, 4), "):", n_influential, "\n")
if (n_influential > 0) {
  top_k <- head(df_ck[order(-df_ck$cook), ], 10)
  cat("Top influential indices (up to 10):", paste(top_k$idx, collapse = ", "), "\n")
}

cat("\n=== Interpretation & Remedies ===\n")
cat("- Residual plot：若局部平滑线明显偏离 0 或呈扇形，存在非线性或异方差。\n")
cat("- Q–Q plot：尾部偏离较大说明残差非正态，点预测影响小，但区间与显著性受影响。\n")
cat("- Cook's D：超过阈值的观测为影响点，应检查异常或录入错误。\n")

# 7) 违例处理建议（自动判断给出动作）
if (bp$p.value < 0.05) {
  cat("\n[Action] 检测到异方差：建议使用稳健标准误或加权回归。\n")
  rob_se <- coeftest(best_model, vcov = vcovHC(best_model, type = "HC3"))
  print(rob_se)
} else {
  cat("\n[Action] 未检测到显著异方差：常规 OLS 方差估计可用。\n")
}

if (!is.na(sw_p) && sw_p < 0.05) {
  cat("[Action] 残差偏离正态：大样本下影响有限；用于区间推断时可考虑稳健 SE 或自助法。\n")
}

if (n_influential > 0) {
  cat("[Action] 存在影响点：\n")
  cat("  · 逐条核查异常值与数据质量；\n")
  cat("  · 报价极端可保留但需稳健回归/稳健SE；\n")
  cat("  · 备选：在不破坏业务含义前提下对异常解释变量做截尾/Winsorize。\n")
}

cat("\nDiagnostic plots saved to 'diagnostics/':\n",
    "- residuals_vs_fitted.png\n",
    "- qq_plot.png\n",
    "- cooks_distance.png\n")
# 注意：这些图仅放技术附录，不进展示PPT