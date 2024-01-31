library(ggplot2)
library(dplyr)
library(patchwork)
library(stringr)

setwd("/Users/adi/Documents/Github Repos/P_EPIONCHO-IBM/examples/")


configure_python_data <- function(filePath, isPNC=FALSE) {
    df <- read.csv(filePath)
    returnDf <- df %>% filter(measure == "prevalence") %>%
        mutate(
            mean_mf_prev = rowMeans(.[5:ncol(.)], na.rm=TRUE),
        ) %>% rowwise() %>%
        mutate(
            lb_mf_prev = t.test(c_across(5:ncol(.)))["conf.int"][[1]][1],
            ub_mf_prev = t.test(c_across(5:ncol(.)))["conf.int"][[1]][2]
        ) %>%
        select(year_id, age_start, age_end, mean_mf_prev, lb_mf_prev, ub_mf_prev)
    if(isPNC) {
        returnDf2 <- df %>% filter(measure == "pnc") %>%
            mutate(
                mean_pnc = rowMeans(.[5:ncol(.)], na.rm=TRUE),
            ) %>%
            select(year_id, age_start, age_end, mean_pnc)
        returnDf <- returnDf %>% left_join(returnDf2, by=c("year_id"="year_id", 
                                            "age_start"="age_start", 
                                            "age_end"="age_end"))
    }
    return(returnDf)
}

configure_r_data <- function(folderPath, timeStep=1/366, initialYear = 1894, isPNC=FALSE) {
    total_files <- length(list.files(folderPath))
    mf_prevs <- list()
    pncs <- list()
    pnc_eligibles <- list()
    iter <- 1
    for (file in list.files(folderPath)) {
        tmp_df <- readRDS(paste(folderPath, file, sep=""))
        mf_prevs[[iter]] <- tmp_df[1]
        if(isPNC) {
            mf_prevs[[iter]] <- tmp_df$mf_prev
            pncs[[iter]] <- tmp_df$pnc
            pnc_eligibles[[iter]] <- tmp_df$pnc_eligibles
        }
        iter <- iter + 1
    }
    df <- data.frame(mf_prevs) %>%
        mutate(
            mean_mf_prev = rowMeans(.),
            year_id = (row_number() / (1/(timeStep))) + initialYear
        )
    if(isPNC) {
        df_pnc <- data.frame(pncs) %>%
            mutate(
                mean_pnc = (1-rowMeans(., na.rm=TRUE)),
                year_id = (row_number() / (1/(timeStep))) + initialYear
            )
        df_pnc_eligible <- data.frame(pnc_eligibles) %>%
            mutate(
                mean_pnc_eligible = (1-rowMeans(., na.rm=TRUE)),
                year_id = (row_number() / (1/(timeStep))) + initialYear
            )
        df_pnc <- df_pnc %>% left_join(
            df_pnc_eligible, by=c("year_id"="year_id")
        ) %>% select(year_id, mean_pnc, mean_pnc_eligible)
        df <- df %>% left_join(
            df_pnc, by = c("year_id"="year_id")
        ) %>% select(year_id, mean_mf_prev, mean_pnc, mean_pnc_eligible)
    }
    return(df)
}

python_model_data_1 <- configure_python_data("../test_outputs/python_model_output/testing_CIV0162715440-age_grouped_raw_data_pre-mox-fix.csv", isPNC=TRUE)
python_model_data_2 <- configure_python_data("../test_outputs/python_model_output/testing_CIV0162715440-age_grouped_raw_data_variable_abr_366days_reset_zeros_pnc_updated_dd_daily_sampling.csv", isPNC=TRUE)
python_model_data_3 <- configure_python_data("../test_outputs/python_model_output/testing_CIV0162715440-age_grouped_raw_data_set_abr_366days_reset_zeros_pnc_updated_dd_daily_sampling.csv", isPNC=TRUE)


# timestep = 1/2 a day
r_model_data_1 <- configure_r_data("../test_outputs/output_rank_pnc_0_final/", isPNC=TRUE, timeStep = 0.5/366)
# timestep = 1 day
#r_model_data_1 <- configure_r_data("../test_outputs/output_rank_pnc_0_final/", isPNC=TRUE)
# timestep = 1/2 a day
r_model_data_2 <- configure_r_data("../test_outputs/output_sort_pnc_0_final/", isPNC=TRUE, timeStep = 0.5/366)
# timestep = 1 day
#r_model_data_2 <- configure_r_data("../test_outputs/output_sort_pnc_0_final_old/", isPNC=TRUE)



mfp_plot <- ggplot() +
    geom_line(aes(x=year_id, y=mean_mf_prev, color="Original R Model"), data=r_model_data_1) +
    scale_color_manual("Model", values=c(
                        "Original R Model"="grey",
                        "Fixed R Model"="black",
                        "Original Python Model Variable kE/ABR"="blue",
                        "Fixed Python Model Variable kE/ABR" = "red",
                        "Fixed Python Model Constant kE/ABR" = "purple"
                        ),
                        labels = function(x) str_wrap(x, width=16)) +
    scale_y_continuous("MF Prevalence (%)", limits=c(0, 0.8), breaks=seq(0, 1, 0.1)) +
    geom_vline(aes(xintercept=1988), color="gold", linetype="dashed") +
    annotate("text", color="gold", label="52% IVM Treatment Starts", x=1988, y=0.1, hjust=1) +
    geom_vline(aes(xintercept=1997), color="orange", linetype="dashed") +
    annotate("text", color="orange",label="65% IVM Treatment Starts", x=1997, y=0.75, hjust=1) +
    geom_vline(aes(xintercept=2000), color="maroon", linetype="dashed") +
    annotate("text", color="maroon",label="Treatment Stops", x=2000, y=0.65, hjust=1) +
    geom_vline(aes(xintercept=2026), color="darkred", linetype="dashed") +
    annotate("text", color="darkred", label="65% MOX Treatment Starts", x=2026, y=0.7, hjust=1) +
    theme_bw() +
    guides(color=guide_legend(
        label.hjust=1
    )) +
    scale_x_continuous("Time (years)", limits=c(1970, 2030), breaks=seq(1900, 2040, 10))

pnc_plot <- ggplot() +
    geom_line(aes(x=year_id, y=mean_pnc_eligible, color="Original R Model"), alpha=1, data=r_model_data_1) +
    theme_bw() +
    scale_color_manual("Model", values=c(
                        "Original R Model"="grey",
                        "Fixed R Model"="black",
                        "Original Python Model Variable kE/ABR"="blue",
                        "Fixed Python Model Variable kE/ABR" = "red",
                        "Fixed Python Model Constant kE/ABR" = "purple"
                        ),
                        labels = function(x) str_wrap(x, width=16)) +
    scale_y_continuous("PNC (%)", limits=c(0, 0.8), breaks=seq(0, 1, 0.1)) +
    geom_vline(aes(xintercept=1988), color="gold", linetype="dashed") +
    annotate("text", color="gold", label="52% IVM Treatment Starts", x=1988, y=0.1, hjust=1) +
    geom_vline(aes(xintercept=1997), color="orange", linetype="dashed") +
    annotate("text", color="orange",label="65% IVM Treatment Starts", x=1997, y=0.75, hjust=1) +
    geom_vline(aes(xintercept=2000), color="maroon", linetype="dashed") +
    annotate("text", color="maroon",label="Treatment Stops", x=2000, y=0.65, hjust=1) +
    geom_vline(aes(xintercept=2026), color="darkred", linetype="dashed") +
    annotate("text", color="darkred", label="65% MOX Treatment Starts", x=2026, y=0.7, hjust=1) +
    guides(color=guide_legend(
        label.hjust=1
    )) +
    scale_x_continuous("Time (years)", limits=c(1970, 2030), breaks=seq(1900, 2040, 10))

ggsave("../test_outputs/comparison_plots1.png", (mfp_plot / pnc_plot), 
    dpi=800, width=8000, height=4000, units="px")

mfp_plot <- mfp_plot + geom_line(aes(x=year_id, y=mean_mf_prev, color="Fixed R Model"), alpha=0.7, data=r_model_data_2)
pnc_plot <- pnc_plot + geom_line(aes(x=year_id, y=mean_pnc_eligible, color="Fixed R Model"), alpha=0.7, data=r_model_data_2)

ggsave("../test_outputs/comparison_plots2.png", (mfp_plot / pnc_plot), 
    dpi=800, width=8000, height=4000, units="px")

mfp_plot <- mfp_plot  + geom_line(aes(x=year_id, y=mean_mf_prev, color="Original Python Model Variable kE/ABR"), alpha=0.5, data=python_model_data_1)
pnc_plot <- pnc_plot  + geom_line(aes(x=year_id, y=mean_pnc, color="Original Python Model Variable kE/ABR"), alpha=0.5, data=python_model_data_1)

ggsave("../test_outputs/comparison_plots3.png", (mfp_plot / pnc_plot), 
    dpi=800, width=8000, height=4000, units="px")

mfp_plot <- mfp_plot  + geom_line(aes(x=year_id, y=mean_mf_prev, color="Fixed Python Model Variable kE/ABR"), alpha=0.5, data=python_model_data_2)
pnc_plot <- pnc_plot  + geom_line(aes(x=year_id, y=mean_pnc, color="Fixed Python Model Variable kE/ABR"), alpha=0.5, data=python_model_data_2)

ggsave("../test_outputs/comparison_plots4.png", (mfp_plot / pnc_plot), 
    dpi=800, width=8000, height=4000, units="px")

mfp_plot <- mfp_plot  + geom_line(aes(x=year_id, y=mean_mf_prev, color="Fixed Python Model Constant kE/ABR"), alpha=0.5, data=python_model_data_3)
pnc_plot <- pnc_plot  + geom_line(aes(x=year_id, y=mean_pnc, color="Fixed Python Model Constant kE/ABR"), alpha=0.5, data=python_model_data_3)

ggsave("../test_outputs/comparison_plots5.png", (mfp_plot / pnc_plot), 
    dpi=800, width=8000, height=4000, units="px")


