library(ggplot2)
library(dplyr)
library(patchwork)
library(stringr)

configure_python_data <- function(filePath, isPNC=FALSE) {
    df <- read.csv(filePath)
    returnDf <- df %>% filter(measure == "prevalence") %>%
        mutate(
            mean_mf_prev = rowMeans(.[5:ncol(.)], na.rm=TRUE),
        ) %>%
        select(year_id, age_start, age_end, mean_mf_prev)
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

# only use this if you have raw R data you want to look at
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

python_model_data_1 <- configure_python_data("test_outputs/python_model_output/testing_CIV0162715440-original_model.csv", isPNC=TRUE)
print(head(python_model_data_1))
python_model_data_2 <- configure_python_data("test_outputs/python_model_output/testing_CIV0162715440-new_run.csv", isPNC=TRUE)
print(head(python_model_data_2))


# timestep = 1/2 a day
r_model_data_1 <- read.csv("test_outputs/r_model_output/summarized_model_result.csv")
print(head(r_model_data_1))

mfp_plot <- ggplot() +
    geom_vline(aes(xintercept=1988), color="gold", linetype="dashed") +
    annotate("text", color="gold", label="52% IVM Treatment Starts", x=1988, y=0.1, hjust=1) +
    geom_vline(aes(xintercept=1997), color="orange", linetype="dashed") +
    annotate("text", color="orange",label="65% IVM Treatment Starts", x=1997, y=0.75, hjust=1) +
    geom_vline(aes(xintercept=2000), color="maroon", linetype="dashed") +
    annotate("text", color="maroon",label="Treatment Stops", x=2000, y=0.65, hjust=1) +
    geom_vline(aes(xintercept=2026), color="darkred", linetype="dashed") +
    annotate("text", color="darkred", label="65% MOX Treatment Starts", x=2026, y=0.7, hjust=1) +
    geom_line(aes(x=year_id, y=mean_mf_prev, color="R Model"), data=r_model_data_1) +
    geom_line(aes(x=year_id, y=mean_mf_prev, color="Original Python Model"), alpha=0.5, data=python_model_data_1) +
    geom_line(aes(x=year_id, y=mean_mf_prev, color="New Python Model"), alpha=0.5, data=python_model_data_2) +
    scale_color_manual("Model", values=c(
                        "R Model"="black",
                        "Original Python Model"="blue",
                        "New Python Model" = "red"
                        ),
                        labels = function(x) str_wrap(x, width=16)) +
    scale_y_continuous("MF Prevalence (%)", limits=c(0, 0.8), breaks=seq(0, 1, 0.1)) +
    theme_bw() +
    guides(color=guide_legend(
        label.hjust=1
    )) +
    scale_x_continuous("Time (years)", limits=c(1970, 2030), breaks=seq(1900, 2040, 10))

pnc_plot <- ggplot() +
    geom_vline(aes(xintercept=1988), color="gold", linetype="dashed") +
    annotate("text", color="gold", label="52% IVM Treatment Starts", x=1988, y=0.1, hjust=1) +
    geom_vline(aes(xintercept=1997), color="orange", linetype="dashed") +
    annotate("text", color="orange",label="65% IVM Treatment Starts", x=1997, y=0.75, hjust=1) +
    geom_vline(aes(xintercept=2000), color="maroon", linetype="dashed") +
    annotate("text", color="maroon",label="Treatment Stops", x=2000, y=0.65, hjust=1) +
    geom_vline(aes(xintercept=2026), color="darkred", linetype="dashed") +
    annotate("text", color="darkred", label="65% MOX Treatment Starts", x=2026, y=0.7, hjust=1) +
    geom_line(aes(x=year_id, y=mean_pnc_eligible, color="R Model"), data=r_model_data_1) +
    geom_line(aes(x=year_id, y=mean_pnc, color="Original Python Model"), alpha=0.5, data=python_model_data_1) +
    geom_line(aes(x=year_id, y=mean_pnc, color="New Python Model"), alpha=0.5, data=python_model_data_2) +
    scale_color_manual("Model", values=c(
                        "R Model"="black",
                        "Original Python Model"="blue",
                        "New Python Model" = "red"
                        ),
                        labels = function(x) str_wrap(x, width=16)) +
    scale_y_continuous("PNC (%)", limits=c(0, 0.8), breaks=seq(0, 1, 0.1)) +
    guides(color=guide_legend(
        label.hjust=1
    )) +
    scale_x_continuous("Time (years)", limits=c(1970, 2030), breaks=seq(1900, 2040, 10))

ggsave("test_outputs/comparison_plot.png", (mfp_plot / pnc_plot), 
    dpi=800, width=8000, height=4000, units="px")
