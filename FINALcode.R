# =============================================================================
# DISCRETE CHOICE MODEL — NEIGHBOURHOOD SELECTION, MANCHESTER 2021
# =============================================================================
install.packages("dfidx")
install.packages("zoo")
library(mlogit)
library(dfidx)
library(dplyr)
library(lmtest)
library(ggplot2)

set.seed(42)

setwd("/Users/pranav.hr/Documents/Masters/Topics in Data Science/manchester house prices")

# ---- Load and filter to 2021 ------------------------------------------------
transactions <- read.csv("final_dataset.csv", stringsAsFactors = FALSE)

transactions <- transactions |>
  dplyr::filter(year == 2021) |>
  dplyr::mutate(
    new_build_flag = as.integer(new_build == "Y"),
    buyer_id       = row_number()
  )

text_cols   <- grep("^text_emb",  names(transactions), value = TRUE)  # 25
tfidf_cols  <- grep("^tfidf_emb", names(transactions), value = TRUE)  # 25
struct_cols <- c("avg_price", "price_sd", "n_transactions",
                 "imd_score.y", "crime_score.y", "income",
                 "population", "share_D", "share_F", "share_S")



# =============================================================================
# STEP 1: MSOA-LEVEL ATTRIBUTE TABLE
# =============================================================================

msoa_attrs <- transactions |>
  dplyr::group_by(msoa21cd) |>
  dplyr::summarise(
    across(all_of(c(struct_cols, text_cols, tfidf_cols)), first),
    .groups = "drop"
  )

# --- Standardise raw structural vars (for M1) --------------------------------
for (col in struct_cols) {
  msoa_attrs[[paste0(col, "_z")]] <- as.numeric(scale(msoa_attrs[[col]]))
}
struct_z_cols <- paste0(struct_cols, "_z")

# --- PCA on structural vars (for M2) -----------------------------------------
X_struct   <- scale(as.matrix(msoa_attrs[, struct_cols]))
pca_struct <- prcomp(X_struct, center = FALSE, scale. = FALSE)
cumvar     <- cumsum(pca_struct$sdev^2 / sum(pca_struct$sdev^2))


for (i in 1:5) cat(sprintf("  %d PCs: %.1f%%\n", i, cumvar[i] * 100))




struct_scores          <- as.data.frame(pca_struct$x[, 1:4])
names(struct_scores)   <- paste0("struct_PC", 1:4)
struct_scores$msoa21cd <- msoa_attrs$msoa21cd
msoa_attrs             <- left_join(msoa_attrs, struct_scores, by = "msoa21cd")

struct_pc_cols <- paste0("struct_PC", 1:4)
for (col in struct_pc_cols) {
  msoa_attrs[[col]] <- as.numeric(scale(msoa_attrs[[col]]))
}

# --- Standardise embeddings (for M3 and M4) ----------------------------------
for (col in c(text_cols, tfidf_cols)) {
  msoa_attrs[[col]] <- as.numeric(scale(msoa_attrs[[col]]))
}

# =============================================================================
# BUILD LONG FORMAT
# =============================================================================

all_msoas <- unique(transactions$msoa21cd)

buyers <- transactions |>
  dplyr::select(buyer_id, msoa21cd, new_build_flag) |>
  dplyr::rename(chosen_msoa = msoa21cd)

# All attribute columns needed across all four models
all_attr_cols <- c(struct_z_cols, struct_pc_cols, text_cols, tfidf_cols)

msoa_for_merge <- msoa_attrs |>
  dplyr::select(msoa21cd, all_of(all_attr_cols))

long <- buyers |>
  left_join(data.frame(msoa21cd = all_msoas), by = character()) |>
  dplyr::mutate(chosen = as.integer(msoa21cd == chosen_msoa)) |>
  left_join(msoa_for_merge, by = "msoa21cd") |>
  dplyr::arrange(buyer_id, msoa21cd)

# Build NB interaction for every attribute
nb_flag <- long$new_build_flag
for (col in all_attr_cols) {
  long[[paste0(col, "_nb")]] <- long[[col]] * nb_flag
}

# Build choice_data AFTER all columns exist in long
choice_data <- dfidx(
  long,
  idx    = list("buyer_id", "msoa21cd"),
  choice = "chosen",
  shape  = "long"
)



make_formula <- function(terms) {
  as.formula(paste("chosen ~", paste(terms, collapse = " + "), "| 0"))
}

run_mlogit <- function(formula, data, label) {
  cat(sprintf("Fitting %s ...\n", label))
  tryCatch(
    mlogit(formula, data = data),
    error = function(e) { cat(sprintf("  ERROR: %s\n", e$message)); NULL }
  )
}

# Term vectors — base and NB interaction for each block
# Term vectors — base and NB interaction for each block
m1_base <- struct_z_cols
m1_nb   <- paste0(struct_z_cols, "_nb")

m2_base <- c(struct_pc_cols, "avg_price_z")
m2_nb   <- c(paste0(struct_pc_cols, "_nb"), "avg_price_z_nb")

m3_base <- c(text_cols, "avg_price_z")
m3_nb   <- c(paste0(text_cols, "_nb"), "avg_price_z_nb")

m4_base <- c(tfidf_cols, "avg_price_z")
m4_nb   <- c(tfidf_cols |> paste0("_nb"), "avg_price_z_nb")

# =============================================================================
# MODEL 1: RAW OBSERVED CHARACTERISTICS
# =============================================================================

model_1 <- run_mlogit(make_formula(c(m1_base, m1_nb)), choice_data, "Model 1")
if (!is.null(model_1)) print(summary(model_1))


# =============================================================================
# MODEL 2: STRUCTURAL PCA
# =============================================================================

model_2 <- run_mlogit(make_formula(c(m2_base, m2_nb)), choice_data, "Model 2")
if (!is.null(model_2)) print(summary(model_2))


# =============================================================================
# MODEL 3: SEMANTIC TEXT EMBEDDINGS ONLY
# =============================================================================

model_3 <- run_mlogit(make_formula(c(m3_base, m3_nb)), choice_data, "Model 3")
if (!is.null(model_3)) print(summary(model_3))


# =============================================================================
# MODEL 4: TF-IDF EMBEDDINGS ONLY
# =============================================================================

model_4 <- run_mlogit(make_formula(c(m4_base, m4_nb)), choice_data, "Model 4")
if (!is.null(model_4)) print(summary(model_4))


# =============================================================================
# MODEL COMPARISON — McFadden R² AND FIT STATISTICS
# =============================================================================

models      <- list(model_1, model_2, model_3, model_4)
model_names <- c("M1: Observed (raw)",
                 "M2: Structural PCA",
                 "M3: Text embeddings",
                 "M4: TF-IDF embeddings")

ll_null  <- log(1 / length(all_msoas)) * nrow(buyers)
n_buyers <- nrow(buyers)



for (i in seq_along(models)) {
  m <- models[[i]]
  if (is.null(m)) {
    cat(sprintf("%-22s  FAILED\n", model_names[i]))
    next
  }
  ll   <- as.numeric(logLik(m))
  k    <- length(coef(m))
  rho2 <- round(1 - ll / ll_null, 4)
  aic  <- round(-2 * ll + 2 * k, 1)
  bic  <- round(-2 * ll + k * log(n_buyers), 1)
  cat(sprintf("%-22s  %10.2f  %5d  %8.4f  %10.1f  %10.1f\n",
              model_names[i], ll, k, rho2, aic, bic))
}

# M1 vs M2 (nested — same underlying vars)
if (!is.null(model_1) && !is.null(model_2)) {
  lrt  <- lrtest(model_2, model_1)
  cat(sprintf("M2 vs M1: chi2(%d) = %.2f, p = %.4f\n",
              abs(lrt$Df[2]),
              lrt$Chisq[2],
              lrt$`Pr(>Chisq)`[2]))
}

aic_vals <- sapply(models, function(m) {
  if (is.null(m)) return(Inf)
  ll <- as.numeric(logLik(m))
  k  <- length(coef(m))
  -2 * ll + 2 * k
})
ranking <- order(aic_vals)
for (r in ranking) {
  if (!is.finite(aic_vals[r])) next
  cat(sprintf("  %d. %s  (AIC = %.1f)\n", r, model_names[ranking[r]], aic_vals[ranking[r]]))
}


# =============================================================================
# NEW BUILD PREMIUM TABLE
# =============================================================================

extract_nb_premium <- function(model, base_terms, nb_terms, label) {
  if (is.null(model)) return(invisible(NULL))
  coefs <- coef(model)
  ses   <- sqrt(diag(vcov(model)))
  
  
  for (i in seq_along(base_terms)) {
    b_nm <- base_terms[i]
    g_nm <- nb_terms[i]
    b    <- if (b_nm %in% names(coefs)) coefs[b_nm] else NA
    g    <- if (g_nm %in% names(coefs)) coefs[g_nm] else NA
    if (is.na(b) || is.na(g)) next
    
    p_b <- 2 * pt(abs(b / ses[b_nm]), df = Inf, lower.tail = FALSE)
    p_g <- 2 * pt(abs(g / ses[g_nm]), df = Inf, lower.tail = FALSE)
    
    sig_b <- ifelse(p_b < 0.001, "***", ifelse(p_b < 0.01, "**",
                                               ifelse(p_b < 0.05, "*", "")))
    sig_g <- ifelse(p_g < 0.001, "***", ifelse(p_g < 0.01, "**",
                                               ifelse(p_g < 0.05, "*", "")))
    
    short_nm <- gsub("_z$|_z_nb$", "", b_nm)
    
    cat(sprintf("%-18s  %+6.3f%-3s  %+6.3f%-3s  %+9.4f\n",
                short_nm, b, sig_b, g, sig_g, b + g))
  }
  cat("Significance: *** p<0.001  ** p<0.01  * p<0.05\n")
}

# Call OUTSIDE the function — not nested inside it
extract_nb_premium(model_1, m1_base, m1_nb, "M1: Raw Observed")
extract_nb_premium(model_2, m2_base, m2_nb, "M2: Structural PCA")
extract_nb_premium(model_3, m3_base, m3_nb, "M3: Text Embeddings")
extract_nb_premium(model_4, m4_base, m4_nb, "M4: TF-IDF Embeddings")
# =============================================================================
# VISUALISATIONS — DISCRETE CHOICE MODEL RESULTS
# =============================================================================

library(ggplot2)
library(dplyr)
library(tidyr)

# Colour palette — consistent across all plots
col_resale  <- "#2166ac"   # blue  = resale buyers
col_nb      <- "#d6604d"   # red   = new build buyers
col_m1      <- "#4d4d4d"
col_m2      <- "#878787"
col_m3      <- "#1a9641"
col_m4      <- "#d9ef8b" |> (\(x) "#fdae61")()

theme_dissertation <- theme_minimal(base_size = 12) +
  theme(
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(colour = "grey40", size = 10),
    axis.title    = element_text(size = 11),
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  )


# =============================================================================
# PLOT 1: McFadden R² BAR CHART
# =============================================================================

fit_data <- data.frame(
  model   = c("M1\nRaw Observed", "M2\nStructural PCA",
              "M3\nText Embeddings", "M4\nTF-IDF Embeddings"),
  model_s = c("M1", "M2", "M3", "M4"),
  rho2    = c(0.1235, 0.1165, 0.1320, 0.0989),
  aic     = c(40514,  40818,  40184,  41714),
  bic     = c(40646,  40884,  40528,  42058),
  loglik  = c(-20237, -20399, -20040, -20805),
  k       = c(20, 10, 52, 52)
)

# Ordered by rho2 descending for visual clarity
fit_data <- fit_data |>
  mutate(model = factor(model,
                        levels = fit_data$model[order(fit_data$rho2, decreasing = TRUE)]))

p1 <- ggplot(fit_data, aes(x = model, y = rho2,
                           fill = model_s == "M3")) +
  geom_col(width = 0.6, colour = "white") +
  geom_text(aes(label = sprintf("ρ² = %.3f", rho2)),
            vjust = -0.4, size = 3.5, fontface = "bold") +
  scale_fill_manual(values = c("FALSE" = "#878787", "TRUE" = col_m3),
                    guide = "none") +
  scale_y_continuous(limits = c(0, 0.22),
                     labels = scales::number_format(accuracy = 0.01)) +
  labs(
    x        = NULL,
    y        = "McFadden ρ²"
  ) +
  theme_dissertation

p1

ggsave("plot1_mcfadden_rho2.png", p1, width = 8, height = 5, dpi = 300)
# =============================================================================
# PLOT 2: AIC AND BIC COMPARISON
# =============================================================================

fit_long <- fit_data |>
  select(model, model_s, aic, bic) |>
  pivot_longer(cols = c(aic, bic),
               names_to = "metric", values_to = "value") |>
  mutate(metric = toupper(metric))

p2 <- ggplot(fit_long, aes(x = model, y = value,
                           fill = metric)) +
  geom_col(position = position_dodge(width = 0.7),
           width = 0.6, colour = "white") +
  geom_text(aes(label = scales::comma(round(value))),
            position = position_dodge(width = 0.7),
            vjust = -0.4, size = 3, fontface = "bold") +
  scale_fill_manual(values = c("AIC" = "#4393c3", "BIC" = "#2166ac"),
                    name = "Criterion") +
  scale_y_continuous(labels = scales::comma,
                     limits = c(0, 45000)) +
  labs(
    x        = NULL,
    y        = "Information Criterion"
  ) +
  theme_dissertation

p2

ggsave("plot2_aic_bic.png", p2, width = 9, height = 5, dpi = 300)
# =============================================================================
# PLOT 3: COEFFICIENT DOT PLOT — MODEL 1 RESALE vs NEW BUILD
# =============================================================================

m1_coefs <- data.frame(
  variable  = c("Avg Price", "Price SD", "N Transactions",
                "IMD Score", "Crime Score", "Income",
                "Population", "Share Detached",
                "Share Flats", "Share Semis"),
  beta      = c(0.271783,  0.021787,  0.263343, -0.456915,
                0.154303, -0.415843,  0.022606,  0.099246,
                -0.117960, -0.059048),
  gamma     = c(1.038174, -0.244665, -0.090037,  1.950711,
                -0.527046,  0.659694,  0.558168, -0.622336,
                1.263840, -0.398141),
  sig_beta  = c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE,
                FALSE, TRUE, TRUE, TRUE),
  sig_gamma = c(TRUE, FALSE, TRUE, TRUE, TRUE, TRUE,
                TRUE, TRUE, TRUE, TRUE)
) |>
  mutate(
    nb_total  = beta + gamma,
    variable  = factor(variable,
                       levels = variable[order(beta)])  # sort by resale preference
  )

# Reshape to long for ggplot
m1_long <- m1_coefs |>
  select(variable, beta, nb_total) |>
  pivot_longer(cols = c(beta, nb_total),
               names_to = "buyer_type",
               values_to = "estimate") |>
  mutate(buyer_type = recode(buyer_type,
                             "beta"     = "Resale buyers (β)",
                             "nb_total" = "New build buyers (β + γ)"))

p3 <- ggplot(m1_long,
             aes(x = estimate, y = variable, colour = buyer_type,
                 shape = buyer_type)) +
  geom_vline(xintercept = 0, linetype = "solid",
             colour = "grey70", linewidth = 0.5) +
  geom_line(aes(group = variable),
            colour = "grey70", linewidth = 0.8) +
  geom_point(size = 4) +
  scale_colour_manual(values = c("Resale buyers (β)"         = col_resale,
                                 "New build buyers (β + γ)" = col_nb),
                      name = NULL) +
  scale_shape_manual(values = c("Resale buyers (β)"          = 16,
                                "New build buyers (β + γ)"  = 17),
                     name = NULL) +
  labs(
    x        = "Coefficient estimate",
    y        = NULL
  ) +
  theme_dissertation +
  theme(legend.position = "top")

p3

ggsave("plot3_m1_dot_plot.png", p3, width = 9, height = 6, dpi = 300)
# =============================================================================
# PLOT 4: NEW BUILD PREMIUM HEATMAP — TEXT EMBEDDINGS (M3)
# =============================================================================

text_coefs <- data.frame(
  dim       = paste0("Dim ", 1:25),
  beta      = c(-0.245541, -0.041621,  0.020930,  0.022693, -0.038250,
                -0.049691, -0.093933,  0.078450,  0.152049,  0.053778,
                -0.045053,  0.104521,  0.101063, -0.055587,  0.015405,
                -0.053134,  0.172705, -0.038975,  0.043330,  0.085130,
                0.121182,  0.034468,  0.106449,  0.043932,  0.053539),
  gamma     = c(-1.093594,  4.845004,  0.167689,  0.896349,  1.386135,
                -1.462357, -0.319394, -1.181208,  5.174345,  1.221022,
                1.079398, -2.049671,  1.021772,  0.423255, -1.510884,
                0.367518, -2.630274, -1.064580,  0.633081, -0.327617,
                0.091525,  0.466575,  0.167617, -0.501217,  0.201052)
) |>
  mutate(nb_total = beta + gamma,
         dim      = factor(dim, levels = paste0("Dim ", 1:25)))

text_heatmap <- text_coefs |>
  select(dim, `Resale (β)` = beta, `NB Premium (γ)` = gamma,
         `NB Total (β+γ)` = nb_total) |>
  pivot_longer(-dim, names_to = "type", values_to = "value") |>
  mutate(type = factor(type,
                       levels = c("Resale (β)", "NB Premium (γ)", "NB Total (β+γ)")))

p4 <- ggplot(text_heatmap, aes(x = type, y = dim, fill = value)) +
  geom_tile(colour = "white", linewidth = 0.3) +
  geom_text(aes(label = sprintf("%.2f", value),
                colour = abs(value) > 2),
            size = 2.5) +
  scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#d6604d",
                       midpoint = 0, name = "Estimate",
                       limits = c(-6, 6), oob = scales::squish) +
  scale_colour_manual(values = c("FALSE" = "grey30", "TRUE" = "black"),
                      guide = "none") +
  labs(
    title    = "Text Embedding Coefficients — Model 3",
    subtitle = "NB Premium (γ) values are 10–130× larger than Resale (β) — new build buyers\nrespond far more strongly to Wikipedia-described neighbourhood identity",
    x        = NULL,
    y        = "Embedding Dimension"
  ) +
  theme_dissertation +
  theme(axis.text.x  = element_text(face = "bold"),
        axis.text.y  = element_text(size = 8),
        panel.grid   = element_blank())
p4

ggsave("plot4_text_heatmap.png", p4, width = 8, height = 9, dpi = 300)
# =============================================================================
# PLOT 5: β vs γ SCATTER — TEXT EMBEDDINGS
# =============================================================================
install.packages("ggrepel")
library(ggrepel)

p5 <- ggplot(text_coefs, aes(x = beta, y = gamma, label = dim)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_vline(xintercept = 0, linetype = "dashed", colour = "grey60") +
  geom_point(aes(colour = abs(gamma) > 1), size = 3) +
  ggrepel::geom_text_repel(size = 3, colour = "grey30",
                           max.overlaps = 20) +
  scale_colour_manual(values = c("FALSE" = "grey60", "TRUE" = col_nb),
                      labels = c("FALSE" = "|γ| ≤ 1",
                                 "TRUE"  = "|γ| > 1 (strong NB effect)"),
                      name = NULL) +
  labs(
    title    = "Resale vs New Build Buyer Sensitivity — Text Embeddings",
    x        = "β — Resale buyer preference",
    y        = "γ — Additional new build buyer preference"
  ) +
  theme_dissertation

p5

ggsave("plot5_beta_gamma_scatter.png", p5, width = 8, height = 6, dpi = 300)

# =============================================================================
# PLOT 6: PREDICTED UTILITY — RESALE vs NEW BUILD ACROSS 59 MSOAs
# =============================================================================

# Compute predicted utilities using M1 coefficients
m1_beta_vec <- c(
  avg_price_z      =  0.271783,
  price_sd_z       =  0.021787,
  n_transactions_z =  0.263343,
  imd_score.y_z    = -0.456915,
  crime_score.y_z  =  0.154303,
  income_z         = -0.415843,
  population_z     =  0.022606,
  share_D_z        =  0.099246,
  share_F_z        = -0.117960,
  share_S_z        = -0.059048
)

m1_gamma_vec <- c(
  avg_price_z      =  1.038174,
  price_sd_z       = -0.244665,
  n_transactions_z = -0.090037,
  imd_score.y_z    =  1.950711,
  crime_score.y_z  = -0.527046,
  income_z         =  0.659694,
  population_z     =  0.558168,
  share_D_z        = -0.622336,
  share_F_z        =  1.263840,
  share_S_z        = -0.398141
)

# Standardise MSOA attributes (must match what was used in model)
struct_cols_m1 <- c("avg_price", "price_sd", "n_transactions",
                    "imd_score.y", "crime_score.y", "income",
                    "population", "share_D", "share_F", "share_S")

msoa_z_mat <- msoa_attrs |>
  select(msoa21cd, all_of(struct_cols_m1)) |>
  mutate(across(all_of(struct_cols_m1), ~ as.numeric(scale(.))))

# Rename to match coefficient names
names(msoa_z_mat) <- c("msoa21cd",
                       paste0(struct_cols_m1, "_z") |>
                         gsub("imd_score.y_z", "imd_score.y_z", x = _))

# Compute utility for each MSOA
z_vars <- names(m1_beta_vec)

util_data <- msoa_z_mat |>
  rowwise() |>
  mutate(
    U_resale = sum(m1_beta_vec * c_across(all_of(z_vars))),
    U_nb     = sum((m1_beta_vec + m1_gamma_vec) * c_across(all_of(z_vars)))
  ) |>
  ungroup()

# Label top and bottom MSOAs
util_data <- util_data |>
  mutate(
    label = ifelse(
      rank(-U_nb) <= 5 | rank(U_nb) <= 5 |
        rank(-U_resale) <= 3 | rank(U_resale) <= 3,
      msoa21cd, ""
    ),
    quadrant = case_when(
      U_nb > 0 & U_resale < 0 ~ "NB preferred",
      U_nb < 0 & U_resale > 0 ~ "Resale preferred",
      U_nb > 0 & U_resale > 0 ~ "Both preferred",
      TRUE                     ~ "Both avoided"
    )
  )

p6 <- ggplot(util_data,
             aes(x = U_resale, y = U_nb,
                 colour = quadrant, label = label)) +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", colour = "grey50") +
  geom_hline(yintercept = 0, colour = "grey80") +
  geom_vline(xintercept = 0, colour = "grey80") +
  geom_point(size = 2.5, alpha = 0.8) +
  ggrepel::geom_text_repel(size = 2.8, max.overlaps = 15,
                           colour = "grey20") +
  scale_colour_manual(
    values = c("NB preferred"     = col_nb,
               "Resale preferred" = col_resale,
               "Both preferred"   = "#1a9641",
               "Both avoided"     = "grey60"),
    name = NULL
  ) +
  labs(
    title    = "Predicted MSOA Utility: Resale vs New Build Buyers",
    subtitle = "Model 1 coefficients | Points above diagonal = more attractive to NB buyers\nPoints left of vertical = resale buyers avoid | Points below horizontal = NB buyers avoid",
    x        = "Predicted utility — Resale buyers",
    y        = "Predicted utility — New build buyers"
  ) +
  theme_dissertation

p6

ggsave("plot6_utility_scatter.png", p6, width = 9, height = 7, dpi = 300)


##############################################################################
# PRICE COEF
##############################################################################

library(ggplot2)
library(dplyr)

price_coefs <- data.frame(
  model    = rep(c("M1: Raw Observed", "M2: Structural PCA",
                   "M3: Text Embeddings", "M4: TF-IDF Embeddings"), each = 2),
  buyer    = rep(c("Resale (β)", "New Build (β + γ)"), times = 4),
  estimate = c(
    0.272,  0.272 + 1.038,
    0.073,  0.073 + 2.457,
    -0.012, -0.012 + 0.926,
    0.277,  0.277 + 1.332
  ),
  se = c(
    0.046, 0.169,
    0.055, 0.117,
    0.037, 0.228,
    0.018, 0.113
  ),
  sig = c("***", "***",
          "",    "***",
          "",    "***",
          "***", "***")
) |>
  mutate(
    ci_lo = estimate - 1.96 * se,
    ci_hi = estimate + 1.96 * se,
    label = paste0(sprintf("%.3f", estimate), sig),
    model = factor(model,
                   levels = c("M4: TF-IDF Embeddings", "M3: Text Embeddings",
                              "M2: Structural PCA",    "M1: Raw Observed")),
    buyer = factor(buyer,
                   levels = c("Resale (β)", "New Build (β + γ)"))
  )

p <- ggplot(price_coefs,
            aes(x = model, y = estimate,
                colour = buyer, shape = buyer)) +
  geom_hline(yintercept = 0, linetype = "dashed",
             colour = "grey50", linewidth = 0.5) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                width = 0.15, linewidth = 0.8,
                position = position_dodge(width = 0.5)) +
  geom_point(size = 4,
             position = position_dodge(width = 0.5)) +
  geom_text(
    aes(
      label = label,
      y     = ifelse(buyer == "Resale (β)",
                     ci_lo - 0.12,
                     ci_hi + 0.12),
      hjust = ifelse(buyer == "Resale (β)", 1, 0)
    ),
    position    = position_dodge(width = 0.5),
    size        = 3.2,
    fontface    = "bold",
    show.legend = FALSE
  ) +
  scale_colour_manual(
    values = c("Resale (β)"        = "#2166ac",
               "New Build (β + γ)" = "#d6604d"),
    name = NULL
  ) +
  scale_shape_manual(
    values = c("Resale (β)"        = 16,
               "New Build (β + γ)" = 17),
    name = NULL
  ) +
  scale_y_continuous(limits = c(-0.9, 3.8)) +
  coord_flip() +
  labs(
    x        = NULL,
    y        = "Coefficient estimate"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", size = 13),
    plot.subtitle      = element_text(colour = "grey40", size = 10),
    axis.text.y        = element_text(size = 11),
    legend.position    = "top",
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_blank()
  )
p

ggsave("price_coefficient_plot.png", p, width = 9, height = 5, dpi = 300)
# =============================================================================
# 5-FOLD SPATIAL CROSS VALIDATION — ALL FOUR MODELS
# =============================================================================

library(dplyr)
library(dfidx)
library(mlogit)


# ---- Pre-subset long format to reduce memory per model --------------------
# Only keep columns needed for each model — speeds up dfidx construction

long_m1 <- long |>
  dplyr::select(buyer_id, msoa21cd, chosen,
                all_of(c(m1_base, m1_nb)))

long_m2 <- long |>
  dplyr::select(buyer_id, msoa21cd, chosen,
                all_of(c(m2_base, m2_nb)))

long_m3 <- long |>
  dplyr::select(buyer_id, msoa21cd, chosen,
                all_of(c(m3_base, m3_nb)))

long_m4 <- long |>
  dplyr::select(buyer_id, msoa21cd, chosen,
                all_of(c(m4_base, m4_nb)))

# ---- Assign MSOAs to 5 folds ----------------------------------------------
set.seed(42)
k_folds <- 5

msoa_folds <- tibble(
  msoa21cd = unique(transactions$msoa21cd)
) |>
  mutate(fold = sample(rep(1:k_folds, length.out = n())))

print(table(msoa_folds$fold))


# ---- Core fold function ---------------------------------------------------

run_fold <- function(f, formula, long_subset) {
  
  # MSOAs held out in this fold
  held_out_msoas <- msoa_folds |>
    dplyr::filter(fold == f) |>
    pull(msoa21cd)
  
  # Split buyers
  train_ids <- buyers |>
    dplyr::filter(!chosen_msoa %in% held_out_msoas) |>
    pull(buyer_id)
  
  test_ids <- buyers |>
    dplyr::filter(chosen_msoa %in% held_out_msoas) |>
    pull(buyer_id)
  
  if (length(test_ids) == 0) return(NULL)
  
  # Training long format
  train_long <- long_subset |>
    dplyr::filter(buyer_id %in% train_ids)
  
  # Build dfidx
  train_dfidx <- tryCatch(
    dfidx(train_long,
          idx    = list("buyer_id", "msoa21cd"),
          choice = "chosen",
          shape  = "long"),
    error = function(e) NULL
  )
  if (is.null(train_dfidx)) return(NULL)
  
  # Fit model
  fitted_model <- tryCatch(
    mlogit(formula, data = train_dfidx),
    error = function(e) NULL
  )
  if (is.null(fitted_model)) return(NULL)
  
  # Predict on test buyers
  coefs        <- coef(fitted_model)
  test_long    <- long_subset |>
    dplyr::filter(buyer_id %in% test_ids)
  
  # Keep only formula vars present in test data
  formula_vars <- names(coefs)[names(coefs) %in% names(test_long)]
  if (length(formula_vars) == 0) return(NULL)
  
  # Compute utilities
  X_test <- as.matrix(test_long[, formula_vars, drop = FALSE])
  test_long$pred_utility <- as.numeric(X_test %*% coefs[formula_vars])
  
  # Compute probabilities and ranks per buyer
  results <- test_long |>
    group_by(buyer_id) |>
    mutate(
      # Subtract max for numerical stability before exp
      util_shifted = pred_utility - max(pred_utility),
      exp_util     = exp(util_shifted),
      prob         = exp_util / sum(exp_util),
      rank         = rank(-pred_utility, ties.method = "first")
    ) |>
    summarise(
      actual_rank     = rank[chosen == 1][1],
      correct_top1    = actual_rank == 1,
      correct_top5    = actual_rank <= 5,
      correct_top10   = actual_rank <= 10,
      log_prob_chosen = log(prob[chosen == 1][1] + 1e-10),
      .groups         = "drop"
    )
  
  return(results)
}

# ---- Run CV for all four models -------------------------------------------

cv_models <- list(
  list(label     = "M1: Raw Observed",
       formula   = make_formula(c(m1_base, m1_nb)),
       long_data = long_m1),
  list(label     = "M2: Structural PCA",
       formula   = make_formula(c(m2_base, m2_nb)),
       long_data = long_m2),
  list(label     = "M3: Text Embeddings",
       formula   = make_formula(c(m3_base, m3_nb)),
       long_data = long_m3),
  list(label     = "M4: TF-IDF Embeddings",
       formula   = make_formula(c(m4_base, m4_nb)),
       long_data = long_m4)
)

null_loglik <- log(1 / length(unique(transactions$msoa21cd))) * nrow(buyers)

cv_output <- lapply(cv_models, function(mod) {
  
  # Run all 5 folds
  fold_results <- lapply(1:k_folds, function(f) {
    cat(sprintf("  Fold %d/5 ...\n", f))
    run_fold(f, mod$formula, mod$long_data)
  })
  
  # Remove failed folds
  fold_results <- Filter(Negate(is.null), fold_results)
  
  if (length(fold_results) == 0) {
    cat(sprintf("  WARNING: all folds failed for %s\n\n", mod$label))
    return(NULL)
  }
  
  all_results <- bind_rows(fold_results)
  
  # Aggregate metrics
  hit_1      <- mean(all_results$correct_top1,  na.rm = TRUE)
  hit_5      <- mean(all_results$correct_top5,  na.rm = TRUE)
  hit_10     <- mean(all_results$correct_top10, na.rm = TRUE)
  mean_rank  <- mean(all_results$actual_rank,   na.rm = TRUE)
  cv_loglik  <- sum(all_results$log_prob_chosen, na.rm = TRUE)
  cv_rho2    <- 1 - (cv_loglik / null_loglik)
  
 
  
  list(
    label      = mod$label,
    hit_1      = hit_1,
    hit_5      = hit_5,
    hit_10     = hit_10,
    mean_rank  = mean_rank,
    cv_loglik  = cv_loglik,
    cv_rho2    = cv_rho2,
    n_buyers   = nrow(all_results),
    n_folds_ok = length(fold_results)
  )
})

cv_output <- Filter(Negate(is.null), cv_output)

# ---- Summary table --------------------------------------------------------


# Hit rate table
for (res in cv_output) {
  cat(sprintf("%-22s  %10.4f  %10.4f  %10.4f  %10.1f\n",
              res$label, res$hit_1, res$hit_5, res$hit_10, res$mean_rank))
}

# Fit table


for (res in cv_output) {
  cat(sprintf("%-22s  %12.2f  %10.4f\n",
              res$label, res$cv_loglik, res$cv_rho2))
}

# In-sample vs out-of-sample comparison
insample <- c(
  "M1: Raw Observed"       = 0.1235,
  "M2: Structural PCA"     = 0.1165,
  "M3: Text Embeddings"    = 0.1320,
  "M4: TF-IDF Embeddings"  = 0.0989
)



for (res in cv_output) {
  ins  <- insample[res$label]
  outs <- res$cv_rho2
  drop <- outs - ins
  cat(sprintf("%-22s  %14.4f  %16.4f  %12.4f\n",
              res$label, ins, outs, drop))
}

library(ggplot2)
library(dplyr)
library(tidyr)

# ── Data ──────────────────────────────────────────────────────────────────────
fit_data <- data.frame(
  model    = c("M1: Raw Observed", "M2: Structural PCA",
               "M3: Text Embeddings", "M4: TF-IDF Embeddings"),
  k        = c(20, 10, 52, 52),
  loglik   = c(-20237, -20675, -20049, -21242),
  aic      = c(40514,  40818,  40184,  41714),
  bic      = c(40646,  40884,  40528,  42058),
  rho2_in  = c(0.1235, 0.1165, 0.1320, 0.0989),
  rho2_out = c(-0.0979, -0.0565, -1.0309, -0.4026),
  hit5_out = c(0.015,  0.021,  0.000,  0.000),
  rank_out = c(33.8,   31.3,   41.7,   37.6)
) |>
  mutate(
    drop      = rho2_out - rho2_in,
    model     = factor(model, levels = rev(c(
      "M1: Raw Observed", "M2: Structural PCA",
      "M3: Text Embeddings", "M4: TF-IDF Embeddings")))
  )

# ── Colours ───────────────────────────────────────────────────────────────────
TERRA     <- "#B85042"
SAGE      <- "#A7BEAE"
CHARCOAL  <- "#2C2C2C"
SAND      <- "#E7E8D1"
OFFWHITE  <- "#FAF9F6"
LTGREY    <- "#E0DDD8"
MIDGREY   <- "#9A9490"
WHITE     <- "#FFFFFF"
NEG_COL   <- "#D6604D"   
POS_COL   <- "#1A9641"   
NEUT_COL  <- CHARCOAL

# ── Helper: colour cells by value ────────────────────────────────────────────
cell_colour <- function(val, higher_better = TRUE) {
  if (is.na(val)) return(MIDGREY)
  if (higher_better) {
    ifelse(val >= 0, POS_COL, NEG_COL)
  } else {
    ifelse(val <= 0, POS_COL, NEG_COL)
  }
}

# ── Build long table for ggplot ───────────────────────────────────────────────
table_df <- fit_data |>
  mutate(
    k_lab        = as.character(k),
    loglik_lab   = formatC(loglik,   format = "f", digits = 0, big.mark = ","),
    aic_lab      = formatC(aic,      format = "f", digits = 0, big.mark = ","),
    bic_lab      = formatC(bic,      format = "f", digits = 0, big.mark = ","),
    rho2_in_lab  = sprintf("%.4f", rho2_in),
    rho2_out_lab = sprintf("%.4f", rho2_out),
    drop_lab     = sprintf("%.4f", drop),
    hit5_lab     = sprintf("%.3f", hit5_out),
    rank_lab     = sprintf("%.1f",  rank_out)
  )

# Column definitions: name, label, value col, higher_better
cols <- list(
  list(x = 1,  lab = "k",             col = "k_lab",        hb = FALSE, neutral = TRUE),
  list(x = 2,  lab = "Log-Lik",       col = "loglik_lab",   hb = TRUE,  neutral = TRUE),
  list(x = 3,  lab = "AIC",           col = "aic_lab",      hb = FALSE, neutral = TRUE),
  list(x = 4,  lab = "BIC",           col = "bic_lab",      hb = FALSE, neutral = TRUE),
  list(x = 5,  lab = "In-sample ρ²",  col = "rho2_in_lab",  hb = TRUE,  neutral = FALSE),
  list(x = 6,  lab = "OOS ρ²",        col = "rho2_out_lab", hb = TRUE,  neutral = FALSE),
  list(x = 7,  lab = "Drop",          col = "drop_lab",     hb = TRUE,  neutral = FALSE),
  list(x = 8,  lab = "Top-5 Hit",     col = "hit5_lab",     hb = TRUE,  neutral = FALSE),
  list(x = 9,  lab = "Mean Rank",     col = "rank_lab",     hb = FALSE, neutral = FALSE)
)

n_cols   <- length(cols)
n_models <- nrow(table_df)
row_h    <- 0.7
header_y <- n_models + 0.5
y_vals   <- rev(seq_len(n_models))   # top model = highest y

p <- ggplot() +
  theme_void() +
  theme(
    plot.background  = element_rect(fill = OFFWHITE, colour = NA),
    plot.margin      = margin(20, 20, 20, 20)
  ) +
  coord_cartesian(
    xlim = c(0.4, n_cols + 0.6),
    ylim = c(0.3, n_models + 1.2)
  )

# Vertical divider between in-sample (cols 1-4) and OOS (cols 5-9)
p <- p +
  annotate("segment",
           x = 4.5, xend = 4.5,
           y = 0.3,  yend = n_models + 1.1,
           colour = TERRA, linewidth = 0.8, linetype = "solid")

# Section headers
p <- p +
  annotate("rect",
           xmin = 0.5, xmax = 4.5,
           ymin = n_models + 0.75, ymax = n_models + 1.15,
           fill = CHARCOAL, colour = NA) +
  annotate("text",
           x = 2.5, y = n_models + 0.95,
           label = "In-Sample Fit",
           colour = WHITE, size = 3.8, fontface = "bold", hjust = 0.5) +
  annotate("rect",
           xmin = 4.5, xmax = n_cols + 0.5,
           ymin = n_models + 0.75, ymax = n_models + 1.15,
           fill = TERRA, colour = NA) +
  annotate("text",
           x = 7, y = n_models + 0.95,
           label = "Out-of-Sample (5-Fold Spatial CV)",
           colour = WHITE, size = 3.8, fontface = "bold", hjust = 0.5)

# Individual column headers
for (col in cols) {
  bg <- ifelse(col$x <= 4, CHARCOAL, TERRA)
  p <- p +
    annotate("rect",
             xmin = col$x - 0.5, xmax = col$x + 0.5,
             ymin = n_models + 0.35, ymax = n_models + 0.75,
             fill = bg, colour = LTGREY, linewidth = 0.2) +
    annotate("text",
             x = col$x, y = n_models + 0.55,
             label = col$lab,
             colour = WHITE, size = 3.0, fontface = "bold", hjust = 0.5)
}

# Row label header
p <- p +
  annotate("rect",
           xmin = -0.1, xmax = 0.45,
           ymin = n_models + 0.35, ymax = n_models + 1.15,
           fill = CHARCOAL, colour = NA) +
  annotate("text",
           x = 0.18, y = n_models + 0.75,
           label = "Model",
           colour = WHITE, size = 3.5, fontface = "bold", hjust = 0.5)

# ── Data rows ─────────────────────────────────────────────────────────────────
for (i in seq_len(n_models)) {
  row   <- table_df[i, ]
  yi    <- y_vals[i]
  bg_row <- ifelse(i %% 2 == 0, SAND, WHITE)
  
  # Row label background
  p <- p +
    annotate("rect",
             xmin = -0.1, xmax = 0.45,
             ymin = yi - 0.42, ymax = yi + 0.42,
             fill = CHARCOAL, colour = NA) +
    annotate("text",
             x = 0.18, y = yi,
             label = as.character(row$model),
             colour = WHITE, size = 3.0, fontface = "bold",
             hjust = 0.5, vjust = 0.5)
  
  # Data cells
  for (col in cols) {
    val_raw <- row[[col$col]]
    val_num <- suppressWarnings(as.numeric(gsub(",", "", val_raw)))
    
    # Cell background
    p <- p +
      annotate("rect",
               xmin = col$x - 0.5, xmax = col$x + 0.5,
               ymin = yi - 0.42,   ymax = yi + 0.42,
               fill = bg_row, colour = LTGREY, linewidth = 0.2)
    
    # Text colour
    txt_col <- if (col$neutral) {
      CHARCOAL
    } else {
      if (!is.na(val_num)) cell_colour(val_num, col$hb) else MIDGREY
    }
    
    # Bold if best in column
    is_best <- FALSE
    if (!col$neutral && !is.na(val_num)) {
      all_vals <- suppressWarnings(
        as.numeric(gsub(",", "", table_df[[col$col]])))
      best_val <- if (col$hb) max(all_vals, na.rm = TRUE) else
        min(all_vals, na.rm = TRUE)
      is_best  <- abs(val_num - best_val) < 1e-6
    }
    
    p <- p +
      annotate("text",
               x = col$x, y = yi,
               label = val_raw,
               colour = txt_col,
               size   = 3.1,
               fontface = ifelse(is_best, "bold", "plain"),
               hjust = 0.5, vjust = 0.5)
  }
}

# ── Footer note ───────────────────────────────────────────────────────────────
p <- p +
  annotate("text",
           x = 0.5, y = 0.15,
           label = paste0(
             "Null log-lik = -23,088  |  ",
             "OOS = out-of-sample  |  ",
             "Drop = OOS ρ² − in-sample ρ²  |  ",
             "Top-5 Hit = % buyers where correct MSOA in top 5 predictions  |  ",
             "Mean Rank = avg rank of correct MSOA (random baseline = 30/59)  |  ",
             "Green = best in column   Red = worst in column"
           ),
           colour = MIDGREY, size = 2.4, hjust = 0, vjust = 0)

# Price coefficient plot

p <- p +
  annotate("text",
           x = (n_cols + 1) / 2, y = n_models + 1.35,
           label = "Model Fit: In-Sample vs Out-of-Sample (5-Fold Spatial CV)",
           colour = CHARCOAL, size = 4.5, fontface = "bold", hjust = 0.5)

ggsave("fit_table_plot.png", p,
       width = 14, height = 4.5, dpi = 300, bg = OFFWHITE)

p

library(ggplot2)
library(dplyr)
library(tidyr)

# ── Extract cumulative variance from each PCA object ─────────────────────────
# RUN TF_IDF embeddings and text embeddings R script before this!!!!!!!!!!

struct_cumvar <- cumsum(pca_struct$sdev^2 / sum(pca_struct$sdev^2)) * 100
text_cumvar   <- cumsum(summary(text_pca)$importance[2, ]) * 100
tfidf_cumvar  <- cumsum(summary(tfidf_pca)$importance[2, ]) * 100

# ── Build data frames — limit to first 50 PCs for text/tfidf ─────────────────
struct_df <- data.frame(
  pc     = 1:length(struct_cumvar),
  cumvar = struct_cumvar,
  model  = "M2: Structural PCA"
)

text_df <- data.frame(
  pc     = 1:min(50, length(text_cumvar)),
  cumvar = text_cumvar[1:min(50, length(text_cumvar))],
  model  = "M3: Text Embeddings"
)

tfidf_df <- data.frame(
  pc     = 1:min(50, length(tfidf_cumvar)),
  cumvar = tfidf_cumvar[1:min(50, length(tfidf_cumvar))],
  model  = "M4: TF-IDF Embeddings"
)

# ── Combine ───────────────────────────────────────────────────────────────────
all_pca <- bind_rows(struct_df, text_df, tfidf_df) |>
  mutate(model = factor(model, levels = c(
    "M2: Structural PCA",
    "M3: Text Embeddings",
    "M4: TF-IDF Embeddings"
  )))

# ── Variance captured at retained PCs ────────────────────────────────────────
retained_df <- data.frame(
  model = factor(c("M2: Structural PCA",
                   "M3: Text Embeddings",
                   "M4: TF-IDF Embeddings"),
                 levels = c("M2: Structural PCA",
                            "M3: Text Embeddings",
                            "M4: TF-IDF Embeddings")),
  pc_retained = c(4, 25, 25),
  cumvar_at_retained = c(
    struct_cumvar[4],
    text_cumvar[25],
    tfidf_cumvar[25]
  )
)

# ── Plot ──────────────────────────────────────────────────────────────────────
p_pca <- ggplot(all_pca, aes(x = pc, y = cumvar)) +
  
  geom_hline(yintercept = 80, linetype = "dashed",
             colour = "grey50", linewidth = 0.5) +
  
  geom_line(colour = "#1a9641", linewidth = 1) +
  geom_point(colour = "#1a9641", size = 1.8) +
  
  geom_vline(data = retained_df,
             aes(xintercept = pc_retained),
             linetype = "dotted", colour = "#d6604d",
             linewidth = 0.8) +
  
  geom_text(data = retained_df,
            aes(x     = pc_retained + 0.5,
                y     = 15,
                label = paste0(pc_retained, " PCs\n(",
                               round(cumvar_at_retained, 1), "%)")),
            hjust = 0, size = 3, colour = "#d6604d") +
  
  annotate("text", x = 0.5, y = 82,
           label = "80% threshold",
           hjust = 0, size = 3, colour = "grey40") +
  
  facet_wrap(~ model, scales = "free_x", ncol = 3) +
  
  scale_y_continuous(limits = c(0, 100),
                     breaks = seq(0, 100, 20),
                     labels = function(x) paste0(x, "%")) +
  
  scale_x_continuous(breaks  = function(x) pretty(x, n = 5),
                     expand  = expansion(mult = c(0.05, 0.35))) +
  
  labs(
    x        = "Number of principal components",
    y        = "Cumulative variance explained"
  ) +
  
  theme_minimal(base_size = 12) +
  theme(
    plot.title       = element_text(face = "bold", size = 13),
    plot.subtitle    = element_text(colour = "grey40", size = 10),
    strip.text       = element_text(face = "bold", size = 11),
    panel.grid.minor = element_blank(),
    plot.margin      = margin(12, 16, 12, 12)
  )

p_pca
ggsave("appendix_pca_variance.png", p_pca,
       width = 12, height = 4.5, dpi = 300)
cat("Saved: appendix_pca_variance.png\n")

# =============================================================================
# IMD COEFFICIENT PLOT
# =============================================================================

get_coef <- function(model, name) {
  coefs <- coef(model)
  ses   <- sqrt(diag(vcov(model)))
  if (!name %in% names(coefs)) return(list(est = NA, se = NA))
  list(est = coefs[name], se = ses[name])
}

extract_imd <- function(model, label) {
  b     <- get_coef(model, "imd_score.y_z")
  g     <- get_coef(model, "imd_score.y_z_nb")
  if (is.na(b$est) || is.na(g$est)) return(NULL)
  
  p_b <- 2 * pnorm(-abs(b$est / b$se))
  p_g <- 2 * pnorm(-abs((b$est + g$est) / sqrt(b$se^2 + g$se^2)))
  
  sig <- function(p) case_when(p < 0.001 ~ "***", p < 0.01 ~ "**",
                               p < 0.05 ~ "*", TRUE ~ "")
  
  data.frame(
    model    = label,
    buyer    = c("Resale (\u03b2)", "New Build (\u03b2 + \u03b3)"),
    estimate = c(b$est, b$est + g$est),
    se       = c(b$se, sqrt(b$se^2 + g$se^2)),
    sig      = c(sig(p_b), sig(p_g))
  )
}

imd_coefs <- bind_rows(
  extract_imd(model_1, "M1: Raw Observed"),
  extract_imd(model_2, "M2: Structural PCA"),
  extract_imd(model_3, "M3: Text Embeddings"),
  extract_imd(model_4, "M4: TF-IDF Embeddings")
) |>
  mutate(
    ci_lo = estimate - 1.96 * se,
    ci_hi = estimate + 1.96 * se,
    label = paste0(sprintf("%.3f", estimate), sig),
    model = factor(model,
                   levels = rev(c("M1: Raw Observed",
                                  "M2: Structural PCA",
                                  "M3: Text Embeddings",
                                  "M4: TF-IDF Embeddings"))),
    buyer = factor(buyer,
                   levels = c("Resale (\u03b2)", "New Build (\u03b2 + \u03b3)"))
  )

p_imd <- ggplot(imd_coefs,
                aes(x = model, y = estimate,
                    colour = buyer, shape = buyer)) +
  geom_hline(yintercept = 0, linetype = "dashed",
             colour = "grey50", linewidth = 0.5) +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi),
                width = 0.15, linewidth = 0.8,
                position = position_dodge(width = 0.5)) +
  geom_point(size = 4,
             position = position_dodge(width = 0.5)) +
  geom_text(
    aes(
      label = label,
      y     = ifelse(buyer == "Resale (\u03b2)",
                     ci_lo - 0.12,
                     ci_hi + 0.12),
      hjust = ifelse(buyer == "Resale (\u03b2)", 1, 0)
    ),
    position    = position_dodge(width = 0.5),
    size        = 3.2,
    fontface    = "bold",
    show.legend = FALSE
  ) +
  scale_colour_manual(
    values = c("Resale (\u03b2)"             = "#2166ac",
               "New Build (\u03b2 + \u03b3)" = "#d6604d"),
    name = NULL
  ) +
  scale_shape_manual(
    values = c("Resale (\u03b2)"             = 16,
               "New Build (\u03b2 + \u03b3)" = 17),
    name = NULL
  ) +
  coord_flip() +
  labs(
    x        = NULL,
    y        = "Coefficient estimate"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    plot.title         = element_text(face = "bold", size = 13),
    axis.text.y        = element_text(size = 11),
    legend.position    = "top",
    panel.grid.minor   = element_blank(),
    panel.grid.major.y = element_blank()
  )

p_imd
ggsave("imd_coefficient_plot.png", p_imd, width = 9, height = 5, dpi = 300)


## Summary stats

# Transaction level
trans_stats <- transactions |>
  dplyr::summarise(
    n              = n(),
    new_build_mean = mean(new_build_flag),
    new_build_sd   = sd(new_build_flag),
    new_build_med  = median(new_build_flag)
  )

# MSOA level
msoa_stats <- msoa_attrs |>
  dplyr::summarise(
    across(all_of(struct_cols),
           list(mean   = mean,
                sd     = sd,
                median = median,
                min    = min,
                max    = max),
           .names = "{.col}_{.fn}"))

print(trans_stats)
print(t(msoa_stats))
