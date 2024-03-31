# Minimally adjusted from https://github.com/rvandewater/YAIB-cohorts/
# and https://github.com/prockenschaub/icuDG-preprocessing/
Sys.setenv(RICU_DATA_PATH = 
             "/userdata/lucabri/icu-experiments/data")
Sys.setenv(RICU_CONFIG_PATH = "/u/lucabri/Schreibtisch/icu-experiments/config")

library(argparser)
library(assertthat)
library(rlang)
library(data.table)
library(vctrs)
library(yaml)
library(ricu)

source("src/misc.R")
source("src/steps.R")
source("src/sequential.R")
source("src/obs_time.R")

# Create a parser
p <- arg_parser("Extract and preprocess ICU mortality data")
p <- add_argument(p, "--src", help="source database", default="mimic_demo")
argv <- parse_args(p)
src <- argv$src

out_path <- paste0("/userdata/lucabri/icu-experiments/data/processed_new/", src)

if (!dir.exists(out_path)) {
  dir.create(out_path, recursive = TRUE)
}

cncpt_env <- new.env()

# Task description
time_flow <- "sequential" # sequential / continuous
time_unit <- hours
freq <- 1L
max_len <- 7 * 24  # = 7 days

# On mac, it is advisable to set the R_MAX_VSIZE environment variable before running this.
# E.g., run `R_MAX_VSIZE=64000000000 Rscript base_cohort.R`
static_vars <- c("age", "sex", "height", "weight", "hospital_id", "death", "los_hosp", "los_icu", "insurance", "services", "year", "urgency", "ethnic", "icu_adm_dow", "hosp_adm_dow")

dynamic_vars <- c("alb", "alp", "alt", "ast", "be", "bicar", "bili", "bili_dir",
                  "bnd", "bun", "ca", "cai", "ck", "ckmb", "cl", "crea", "crp",
                  "dbp", "fgn", "fio2", "glu", "hgb", "hr", "inr_pt", "k", "lact",
                  "lymph", "map", "mch", "mchc", "mcv", "methb", "mg", "na", "neut",
                  "o2sat", "pco2", "ph", "phos", "plt", "po2", "ptt", "resp", "sbp",
                  "temp", "tnt", "urine", "wbc")

if (src == "mimic") {
  stay_id = "icustay_id"
} else if (src == "miiv") {
  stay_id = "stay_id"
} else if (src == "eicu") {
  stay_id = "patientunitstayid"
}

if (src == "mimic" || src == "miiv" || src == "eicu") {
  patients <- change_interval(change_id(stay_windows(src, id_type="icustay"), stay_id, src = src), time_unit(freq))
  patients[, start := 0]
} else {
  patients <- stay_windows(src, id_type="icustay", interval = time_unit(freq))
}

patients <- as_win_tbl(patients, index_var = "start", dur_var = "end", interval = time_unit(freq))
arrow::write_parquet(patients, paste0(out_path, "/patients.parquet"))

# Define observation times ------------------------------------------------

stop_obs_at(patients, offset = ricu:::re_time(hours(max_len), time_unit(freq)), by_ref = TRUE)

# Apply exclusion criteria ------------------------------------------------

# 1. Invalid LoS
excl1 <- patients[end < 0, id_vars(patients), with = FALSE]

# 2. Stay <6h
x <- load_step("los_icu")
arrow::write_parquet(x, paste0(out_path, "/los_icu.parquet"))
x <- filter_step(x, ~ . < 6 / 24)

excl2 <- unique(x[, id_vars(x), with = FALSE])

# 3. Less than 4 measurements
n_obs_per_row <- function(x, ...) {
  # TODO: make sure this does not change by reference if a single concept is provided
  obs <- data_vars(x)
  x[, n := as.vector(rowSums(!is.na(.SD))), .SDcols = obs]
  x[, .SD, .SDcols = !c(obs)]
}

x <- load_step(dynamic_vars, interval=time_unit(freq), cache = TRUE)
x <- summary_step(x, "count", drop_index = TRUE)
x <- filter_step(x, ~ . < 4)

excl3 <- unique(x[, id_vars(x), with = FALSE])

# 4. More than 12 hour gaps between measurements
map_to_grid <- function(x) {
  grid <- ricu::expand(patients)
  merge(grid, x, all.x = TRUE)
}

longest_rle <- function(x, val) {
  x <- x[, rle(.SD[[data_var(x)]]), by = c(id_vars(x))]
  x <- x[values != val, lengths := 0]
  x[, .(lengths = max(lengths)), , by = c(id_vars(x))]
}

x <- load_step(dynamic_vars, interval=time_unit(freq), cache = TRUE)
x <- function_step(x, map_to_grid)
x <- function_step(x, n_obs_per_row)
x <- mutate_step(x, ~ . > 0)
x <- function_step(x, longest_rle, val = FALSE)
x <- filter_step(x, ~ . > as.numeric(ricu:::re_time(hours(12), time_unit(1)) / freq))

excl4 <- unique(x[, id_vars(x), with = FALSE])

# 5. Age < 18
# x <- load_step("age")
# x <- filter_step(x, ~ . < 18)

# excl5 <- unique(x[, id_vars(x), with = FALSE])

# 6. LoS less than 48 hours
# x <- load_step("los_icu")
# x <- filter_step(x, ~ . < 48 / 24)

# excl6 <- unique(x[, id_vars(x), with = FALSE])

# Apply exclusions
patients <- exclude(patients, mget(paste0("excl", 1:4)))
attrition <- as.data.table(patients[c("incl_n", "excl_n_total", "excl_n")])
patients <- patients[['incl']]
patient_ids <- patients[, .SD, .SDcols = id_var(patients)]


# Prepare data ------------------------------------------------------------

# Get predictors
dyn <- load_step(dynamic_vars, interval=time_unit(freq), cache = TRUE)
sta <- load_step(static_vars, cache = TRUE)

# Transform all variables into the target format
dyn_fmt <- function_step(dyn, map_to_grid)
rename_cols(dyn_fmt, c("stay_id", "time"), meta_vars(dyn_fmt), by_ref = TRUE)

sta_fmt <- sta[patient_ids]  # TODO: make into step
rename_cols(sta_fmt, c("stay_id"), id_vars(sta), by_ref = TRUE)


# Write to disk -----------------------------------------------------------



arrow::write_parquet(dyn_fmt, paste0(out_path, "/dyn.parquet"))
arrow::write_parquet(sta_fmt, paste0(out_path, "/sta.parquet"))

fwrite(attrition, paste0(out_path, "/attrition.csv"))

if (src %in% c("mimic", "miiv")) {
  caregiver = ricu::load_concepts(c("caregiver", "provider"), src, aggregate = unique)
  arrow::write_parquet(caregiver, paste0(out_path, "/caregiver.parquet"))
  adm_caregiver = ricu::load_concepts(c("adm_caregiver", "adm_provider"), src, aggregate = unique)
  arrow::write_parquet(adm_caregiver, paste0(out_path, "/adm_caregiver.parquet"))
}

if (src == "eicu") {
  arrow::write_parquet(as.data.frame(ricu::eicu$hospital), paste0(out_path, "/hospital.parquet"))
}
if (src == "eicu_demo") {
  arrow::write_parquet(as.data.frame(ricu::eicu$hospital), paste0(out_path, "/hospital.parquet"))
}