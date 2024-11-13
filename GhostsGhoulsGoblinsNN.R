library(tidymodels)
library(tidyverse)
library(vroom)
library(embed)
library(keras)
library(remotes)


# Read in data
train_data <- vroom("./train.csv")
test_data <- vroom("./test.csv")
sample_sub <- vroom("./sample_submission.csv")

# Make ACTION a factor
train_data$type <- as.factor(train_data$type)

# Create Recipe
ggg_recipe <- recipe(type ~ ., data = train_data) %>%
  update_role(id, new_role = "id") %>% 
  step_mutate_at(color, fn = factor) %>% 
  step_dummy(color) %>% 
  step_zv(all_numeric_predictors()) %>% 
  step_range(all_numeric_predictors(), min = 0, max = 1)
prepped_ggg_recipe <- prep(ggg_recipe)
bake(prepped_ggg_recipe, new_data = train_data)

# NB Model
nn_model <- mlp(hidden_units = tune(),
                epochs = 100) %>% 
  set_mode("classification") %>% 
  set_engine("keras")

# Set workflow
nn_wf <- workflow() %>% 
  add_recipe(ggg_recipe) %>% 
  add_model(nn_model)

# Set tuning grid
tuning_grid <- grid_regular(hidden_units(range = c(1, 20)),
                            levels = 5)

# Set number of folds
folds <- vfold_cv(train_data, v = 5, repeats = 1)

# CV
CV_results <- nn_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc, accuracy))

CV_results %>% 
  collect_metrics() %>% 
  filter(.metric == "accuracy") %>% 
  ggplot(aes(x = hidden_units, y = mean)) + geom_line()

+# Select best tuning parameter
best_tune <- CV_results %>% 
  select_best(metric = "roc_auc")
best_tune

# Finalize workflow
nn_final_wf <- nn_wf %>% 
  finalize_workflow(best_tune) %>% 
  fit(data = train_data)

# Generate predictions
nn_preds <- predict(nn_final_wf,
                    new_data = test_data,
                    type = "class")

# Prepare predictions for Kaggle submissions
kaggle_submission <- nn_preds %>% 
  bind_cols(., test_data) %>% 
  rename(type = .pred_class) %>% 
  select(id, type) %>% 
  arrange(id)

# Write the submission to a csv file
vroom_write(x = kaggle_submission,
            file = "./nn_preds.csv", 
            delim = ",")
