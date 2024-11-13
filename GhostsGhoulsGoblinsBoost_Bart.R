library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(discrim)
library(bonsai)
library(lightgbm)

# Read in data
train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")
sample_sub <- vroom("./sample_submission.csv")

# Make ACTION a factor
train_data$type <- as.factor(train_data$type)

# Create Recipe
ggg_recipe <- recipe(type ~ ., data = train_data) %>%
  step_mutate_at(color, fn = factor) %>% 
  step_zv(all_numeric_predictors()) %>% 
  step_normalize(all_numeric_predictors()) %>% 
  step_pca(all_numeric_predictors(), threshold = 0.9)
prepped_ggg_recipe <- prep(ggg_recipe)
bake(prepped_ggg_recipe, new_data = train_data)

# Boosted Trees Model
boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("lightgbm")

# Set workflow
boost_wf <- workflow() %>% 
  add_recipe(ggg_recipe) %>% 
  add_model(boost_model)

# Set tuning grid
tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- boost_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "accuracy")
best_tune

# Finalize workflow
boost_final_wf <- boost_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
boost_preds <- predict(boost_final_wf,
                    new_data = test_data,
                    type = "class")

# Prepare predictions for Kaggle submissions
kaggle_submission <- boost_preds %>% 
  bind_cols(., test_data) %>% 
  rename(type = .pred_class) %>% 
  select(id, type) %>% 
  arrange(id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "./boost_preds.csv", 
            delim = ",")

# BART Model
bart_model <- bart(trees = tune()) %>% 
  set_mode("classification") %>% 
  set_engine("dbarts")

# Set workflow
bart_wf <- workflow() %>% 
  add_recipe(ggg_recipe) %>% 
  add_model(bart_model)

# Set tuning grid
tuning_grid <- grid_regular(trees(),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- bart_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "accuracy")
best_tune

# Finalize workflow
bart_final_wf <- bart_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
bart_preds <- predict(bart_final_wf,
                       new_data = test_data,
                       type = "class")

# Prepare predictions for Kaggle submissions
kaggle_submission <- bart_preds %>% 
  bind_cols(., test_data) %>% 
  rename(type = .pred_class) %>% 
  select(id, type) %>% 
  arrange(id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "./bart_preds.csv", 
            delim = ",")
