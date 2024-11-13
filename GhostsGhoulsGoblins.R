library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(naivebayes)
library(skimr)
library(DataExplorer)
library(rules)
library(themis)

# Kobe Bryant Shot Selection

# Read in data
train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")
sample_sub <- vroom("./sample_submission.csv")

# Make ACTION a factor
train_data$type <- as.factor(train_data$type)

# EDA
glimpse(train_data)
skim(train_data)
plot_intro(train_data)

plot_correlation(train_data)
plot_bar(train_data)
plot_histogram(train_data)

# Create Recipe
ggg_recipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at(color, fn = factor) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_zv(all_numeric_predictors()) %>% 
  step_range(all_numeric_predictors(), min = 0, max = 1) %>% 
  step_smote(all_outcomes(), neighbors = 3)
prepped_ggg_recipe <- prep(ggg_recipe)
bake(prepped_ggg_recipe, new_data = train_data)

# NB Model
nb_model <- naive_Bayes(Laplace = tune(),
                        smoothness = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("naivebayes")

# Set workflow
nb_wf <- workflow() %>% 
  add_recipe(ggg_recipe) %>% 
  add_model(nb_model)

# Set tuning grid
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "roc_auc")
best_tune

# Finalize workflow
nb_final_wf <- nb_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
nb_preds <- predict(nb_final_wf,
                    new_data = test_data,
                    type = "class")

# Prepare predictions for Kaggle submissions
kaggle_submission <- nb_preds %>% 
  bind_cols(., test_data) %>% 
  rename(type = .pred_class) %>% 
  select(id, type) %>% 
  arrange(id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "./nb_preds.csv", 
            delim = ",")

