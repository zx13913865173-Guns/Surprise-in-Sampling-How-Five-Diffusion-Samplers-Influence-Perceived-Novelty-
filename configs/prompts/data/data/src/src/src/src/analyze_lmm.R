# Paper2 linear mixed models
# Run: Rscript src/analyze_lmm.R

library(lme4)
library(lmerTest)
library(performance)
library(emmeans)
library(dplyr)
library(readr)

dir.create("outputs/stats", showWarnings = FALSE, recursive = TRUE)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
student <- read_csv("data/ratings_student.csv", show_col_types = FALSE)
expert  <- read_csv("data/ratings_expert.csv", show_col_types = FALSE)

student$Sampler       <- factor(student$sampler)
student$Prompt        <- factor(student$prompt_id)
student$Image         <- factor(student$image_id)
student$Rater         <- factor(student$rater_id)
student$AblationLevel <- factor(student$ablation_level)

expert$Sampler <- factor(expert$sampler)
expert$Image   <- factor(expert$image_id)
expert$Expert  <- factor(expert$expert_id)

# ------------------------------------------------------------
# M1: Surprise ~ Sampler + (1|Prompt) + (1|Image)
# ------------------------------------------------------------
m1 <- lmer(surprise ~ Sampler + (1 | Prompt) + (1 | Image), data = student)
sink("outputs/stats/M1_surprise_sampler.txt")
print(summary(m1))
print(r2(m1))
print(anova(m1))
print(emmeans(m1, pairwise ~ Sampler, adjust = "bonferroni"))
sink()

# ------------------------------------------------------------
# M2: Surprise ~ Sampler + Clarity_artifact + (1|Prompt) + (1|Image)
# ------------------------------------------------------------
m2 <- lmer(surprise ~ Sampler + clarity_artifact +
             (1 | Prompt) + (1 | Image), data = student)
sink("outputs/stats/M2_surprise_clarity.txt")
print(summary(m2))
print(r2(m2))
print(anova(m2))
print(emmeans(m2, pairwise ~ Sampler, adjust = "bonferroni"))
sink()

# ------------------------------------------------------------
# M3: Surprise ~ AblationLevel + (1|Prompt) + (1|Image)
# ------------------------------------------------------------
m3 <- lmer(surprise ~ AblationLevel + (1 | Prompt) + (1 | Image), data = student)
sink("outputs/stats/M3_ablation.txt")
print(summary(m3))
print(r2(m3))
print(anova(m3))
print(emmeans(m3, pairwise ~ AblationLevel, adjust = "bonferroni"))
sink()

# ------------------------------------------------------------
# M4: SD3.5 Surprise ~ Sampler + (1|Prompt) + (1|Image)
# ------------------------------------------------------------
sd35 <- student %>% filter(model == "sd35")
if (nrow(sd35) > 0) {
  sd35$Sampler <- factor(sd35$sampler)
  sd35$Prompt  <- factor(sd35$prompt_id)
  sd35$Image   <- factor(sd35$image_id)
  m4 <- lmer(surprise ~ Sampler + (1 | Prompt) + (1 | Image), data = sd35)
  sink("outputs/stats/M4_sd35.txt")
  print(summary(m4))
  print(r2(m4))
  print(anova(m4))
  print(emmeans(m4, pairwise ~ Sampler, adjust = "bonferroni"))
  sink()
}

# ------------------------------------------------------------
# M5: ExpertValue ~ Sampler + (1|Image) + (1|Expert)
# ------------------------------------------------------------
m5 <- lmer(creative_value_composite ~ Sampler +
             (1 | Image) + (1 | Expert), data = expert)
sink("outputs/stats/M5_expert_value.txt")
print(summary(m5))
print(r2(m5))
print(anova(m5))
print(emmeans(m5, pairwise ~ Sampler, adjust = "bonferroni"))
sink()

# ------------------------------------------------------------
# Correlations with BH correction
# ------------------------------------------------------------
img_level <- student %>%
  group_by(image_id) %>%
  summarise(surprise = mean(surprise, na.rm = TRUE), .groups = "drop") %>%
  inner_join(
    expert %>% group_by(image_id) %>%
      summarise(expert_value = mean(creative_value_composite, na.rm = TRUE),
                .groups = "drop"),
    by = "image_id"
  )

cor_test <- cor.test(img_level$surprise, img_level$expert_value)
sink("outputs/stats/correlation_surprise_expert.txt")
print(cor_test)
sink()

# Benjamini-Hochberg adjustment for the four reported correlations
pvals <- c(0.03, 0.004, 0.02, 0.010)
bh <- p.adjust(pvals, method = "BH")
write.csv(data.frame(p = pvals, p_BH = bh),
          "outputs/stats/BH_adjusted_pvalues.csv", row.names = FALSE)

cat("LMM analyses complete. See outputs/stats/.\n")
