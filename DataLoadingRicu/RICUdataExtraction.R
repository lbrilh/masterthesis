# Concepts refer to medical terms like disease, treatment, age,...
# ts_table --> Table with time stamps


# Package installation
install.packages("ricu")
install.packages("arrow")
install.packages("fstcore")
# install.packages(c("mimic.demo", "eicu.demo"),
#                 repos = "https://eth-mds.github.io/physionet-demo")

require(ricu)
require(arrow)

# Set environment variables for data and configuration paths
Sys.setenv(RICU_DATA_PATH = 
             "/userdata/lucabri/icu-experiments/data")
# Sys.setenv(TEMP="/userdata/lucabri/tmp") # Setzen befor R gestartet wird

setwd("/scratch/userdata/lucabri")

# Credentials for PhysioNet Account
Sys.setenv(RICU_PHYSIONET_USER = "Your_Username")
Sys.setenv(RICU_PHYSIONET_PASS = "Your_Password")

# Check dataset availability
ricu::src_data_avail()

static_variables <- c("age", "sex", "height", "weight")

dynamic_variables <- c("alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir",
                       "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl", "crea", "crp",
                       "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact",
                       "lymph", "map", "mch", "mchc", "mcv", "methb", "mg", "na", "neut",
                       "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp", "sbp",
                       "temp", "tnt", "urine", "wbc")

sources <- c("eicu", "mimic", "miiv", "hirid")

# dummy <-- ricu::load_concepts(dynamic_variables,"eicu_demo")

# Call and save ICU data
for (source in sources){
  ricu::src_data_avail()
  cat("Setting up ", source, "\n")
  ricu::setup_src_data(source)
}
ricu::src_data_avail()

# save dynamic data in local directory  
for (source in sources){
  
  # Specify the target file path (including the file name and .parquet extension)
  target =  sprintf("processed/%s_static.parquet", source)
  target_file <- file.path(Sys.getenv("RICU_DATA_PATH"), target)
  if (!file.exists(target_file)) {
    static_data <- ricu::load_concepts(static_variables, c(source))
    arrow::write_parquet(static_data, target_file)
    cat("Static variables from ", source , " saved successfully.\n")
  }
  
  
  # Specify the target file path (including the file name and .parquet extension)
  target =  sprintf("processed/%s_dynamic.parquet", source)
  target_file <- file.path(Sys.getenv("RICU_DATA_PATH"), target)
  if(!file.exists(target_file)){
    dynamic_data <- ricu::load_concepts(dynamic_variables, c(source))
    arrow::write_parquet(dynamic_data, target_file)
    cat("Dynamic variables from ", source, " saved successfully.\n")
  }
  if (source == "eicu") {
    target =  sprintf("data/processed/%s_hospital.parquet", source)
    if (!file.exists(target)) {
      arrow::write_parquet(as.data.frame(ricu::eicu$hospital), target)
    }
  }
  cat("Finished loading files from ", source)
}