import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve,precision_score,recall_score
from sklearn.metrics import roc_auc_score



def return_pos_stage_data(df_admission, df_patient, staging_df):
    guarantee_codes = [
        554,
        555,
        556,
        557,
        558,
        559,
        560,
        561,
        562,
        563,
        564,
        565,
        566,
        567,
        568,
        569,
        570,
        571,
        572,
        573,
        574,
        575,
        576,
        577,
        8457,
        8458,
        8459,
        8460,
        8461,
        8462,
        8463,
        8464,
        8465,
        228506,
        228507,
        228508,
        228509,
        228510,
        228511,
        228512,
        228513,
        228514,
        228515,
        228539,
        228540,
        228541,
        228542,
        228543,
        228544,
        228545,
        228546,
        228547,
        228548,
        228549,
        228550,
        228551,
        228552,
        228553,
        228554,
        228555,
        228556,
        228557,
        228558,
        228610,
        228611,
        228612,
        228613,
        228614,
        228615,
        228616,
        228617,
        228618,
        228619,
        228620,
        228621,
        228622,
        228623,
        228624,
        228625,
        228626,
        228627,
        228628,
        228629,
    ]

    # Adding patient information to ddmission table
    df_patient = df_patient[["SUBJECT_ID", "DOB", "GENDER"]]
    df_admission = df_admission.merge(df_patient, how="inner", on="SUBJECT_ID")
    # Find the first admission time for each patient
    df_age_min = (
        df_admission[["SUBJECT_ID", "ADMITTIME"]]
        .groupby("SUBJECT_ID")
        .min()
        .reset_index()
    )
    df_age_min.columns = ["SUBJECT_ID", "ADMIT_MIN"]
    df_admission = df_admission.merge(df_age_min, how="outer", on="SUBJECT_ID")
    df_admission["DOB"] = pd.to_datetime(df_admission["DOB"]).dt.date
    df_admission["ADMIT_MIN"] = pd.to_datetime(df_admission["ADMIT_MIN"]).dt.date
    df_admission["age"] = df_admission.apply(
        lambda e: ((e["ADMIT_MIN"] - e["DOB"]).days / 365.25), axis=1
    )
    # People who have age <0 are most likely emergency cases who were already dead, set them to default of 90
    #     df_admission["age"] = np.where(df_admission["age"] < 0, 90, df_admission["age"])
    df_admission["age"] = np.where(df_admission["age"] >= 300, 90, df_admission["age"])
    # Removing kids
    df_admission = df_admission[df_admission["age"] >= 15.0]
    df_adm_time = df_admission[["ADMITTIME", "HADM_ID"]]
    care_items = [551, 552, 553]
    meta_items = [
        224631,
        224965,
        224966,
        224967,
        224968,
        224969,
        224970,
        224971,
        227618,
        227619,
    ]
    care_items += guarantee_codes
#     first_stage = []

    first_stage = [
#         "Intact,Color Chg",
#         "Red, Unbroken",
#         "Red; unbroken",
#         "Unable to Stage",
#         "Other/Remarks",
    ]
    
#         "Unable to assess; dressing not removed",
#         "Unable to stage; wound is covered with eschar",
#         "Deep tissue injury",

    not_case_vals = ["Not applicable", "None", "Negative", "No"]
    not_case_vals += first_stage

    care_data = staging_df[staging_df["itemid"].isin(care_items)]
    meta_data = staging_df[staging_df["itemid"].isin(meta_items)]
    care_vals = list(care_data["value"].unique())
    meta_vals = list(meta_data["value"].unique())
    case_vals = [i for i in care_vals + meta_vals if i not in not_case_vals]
    staging_df = staging_df.dropna(subset=["value"])
    staging_df = staging_df.dropna(subset=["charttime"])
    staging_df = staging_df.sort_values(by="charttime")
    
#     old way
#     staging_df = staging_df[staging_df["value"].isin(case_vals)]
#     first_egem = (
#         staging_df.groupby("hadm_id")[["charttime", "value"]].first().reset_index()
#     )

#     new_way
    first_egem = (
        staging_df.groupby("hadm_id")[["charttime", "value"]].first().reset_index()
    )
    first_egem = first_egem[first_egem["value"].isin(case_vals)]

    first_egem_adm = df_adm_time.merge(
        first_egem, left_on="HADM_ID", right_on="hadm_id"
    )
    first_egem_adm["charttime"] = pd.to_datetime(first_egem_adm["charttime"])
    first_egem_adm["ADMITTIME"] = pd.to_datetime(first_egem_adm["ADMITTIME"])
    first_egem_adm["time_to_first_stage"] = first_egem_adm.apply(
        lambda e: ((e["charttime"] - e["ADMITTIME"]).total_seconds() / float(3600)),
        axis=1,
    )
    first_egem_adm_after_one_day = first_egem_adm[
        first_egem_adm["time_to_first_stage"] >= 24
    ]
    case_data_after_one_day = first_egem_adm_after_one_day
    list_stages = (
        case_data_after_one_day.groupby("hadm_id")["value"]
        .apply(set)
        .apply(list)
        .reset_index(name="list_stage_values")
    )
    list_stages["number_of_stagings"] = list_stages["list_stage_values"].apply(len)
    list_stages_hadms = list(list_stages["hadm_id"].values)

    return list_stages_hadms


def marking_PU_in_notes(df_note, df_admission):
    """
    Using keywords and ICD9_CODEs to mark pressure ulcers in each admission
    :param df_note: dataframe of the note table of MIMIC III
    :return: PU marked dataframe
    """

    #     bed sore, bed ulcer, pressure sore, pressure ulcer, decub* sore, decub* ulcer

    #     patterns = bedsore($|[^:]) decub\w* sore($|[^:])
    #     (\w*\s*) sore($|[^:])
    patterns = [
        "bed sore($|[^:])",
        "bed ulcer($|[^:])",
        "pressure sore($|[^:])",
        "pressure ulcer($|[^:])",
        "decub(\w*\s*) sore($|[^:])",
        "decub(\w*\s*) ulcer($|[^:])",
    ]

    df_note = df_note[["HADM_ID", "TEXT", "CHARTDATE", "CHARTTIME"]]
    df_note["TEXT"] = df_note["TEXT"].str.lower()
    all_notes_pu_mention = np.logical_or.reduce(
        [df_note["TEXT"].str.contains(word) for word in patterns]
    )
    df_note["PU_mention_notes"] = pd.DataFrame(
        data=all_notes_pu_mention, index=df_note.index
    )
    df_note.drop(columns=["TEXT"], inplace=True)

    pos_note = df_note[df_note["PU_mention_notes"] > 0]
    df_admission_lim = df_admission[["HADM_ID", "ADMITTIME"]]
    # pos_note['CHARTTIME']  = pos_note['CHARTTIME'].fillna('0000-01-01 00:00:00')
    pos_note["CHARTTIME"] = pd.to_datetime(pos_note["CHARTTIME"])
    pos_note["CHARTDATE"] = pd.to_datetime(pos_note["CHARTDATE"])
    pos_note["CHARTTIME"] = pos_note["CHARTTIME"].fillna(
        pos_note["CHARTDATE"] - pd.Timedelta(days=0)
    )
    pos_note_by_hadm = (
        pos_note.groupby(["HADM_ID"])["CHARTDATE", "CHARTTIME"].min().reset_index()
    )

    pos_note_by_hadm = pos_note_by_hadm.merge(df_admission_lim, on="HADM_ID")

    pos_note_by_hadm["ADMITTIME"] = pd.to_datetime(pos_note_by_hadm["ADMITTIME"])
    pos_note_by_hadm["CHARTDATE"] = pd.to_datetime(pos_note_by_hadm["CHARTDATE"])
    pos_note_by_hadm["CHARTTIME"] = pd.to_datetime(pos_note_by_hadm["CHARTTIME"])

    pos_note_by_hadm["time_to_first_keyw_app"] = pos_note_by_hadm.apply(
        lambda e: ((e["CHARTDATE"] - e["ADMITTIME"]).total_seconds() / float(3600)),
        axis=1,
    )
    pos_note_by_hadm["time_to_first_keyw_app_acc"] = pos_note_by_hadm.apply(
        lambda e: ((e["CHARTTIME"] - e["ADMITTIME"]).total_seconds() / float(3600)),
        axis=1,
    )
    pos_note_by_hadm = pos_note_by_hadm[
        ((pos_note_by_hadm["time_to_first_keyw_app_acc"] >= 24))
    ]

    pos_notes_hadms_after48 = list(pos_note_by_hadm["HADM_ID"].values)

    return pos_notes_hadms_after48


def final_admission_cleaning(df_admission):
    """
    Removing negative length of stays records, and dummifying :D  categorical features
    :param df_admission: dataframe of admission table of MIMIC III
    :return: cleaned admission dataframe
    """
    # remove dead per cms guideline
    df_admission = df_admission[df_admission['DECEASED'] == 0]


    # Remove LOS with negative number, likely entry form error
    df_admission = df_admission[df_admission["LOS_total"] > 0]
    # Drop unused or no longer needed columns keeping HADM ID
    df_admission.drop(
        columns=[
            "SUBJECT_ID",
            "ADMITTIME",
            "ADMISSION_LOCATION",
            "DISCHARGE_LOCATION",
            "LANGUAGE",
            "ADMIT_MIN",
            "DOB",
            "DIAGNOSIS",
            "DECEASED",
            "DEATHTIME",
        ],
        inplace=True,
    )
    # Create dummy columns for categorical variables
    prefix_cols = ["ADM", "INS", "REL", "ETH", "MAR", "GEN"]
    dummy_cols = [
        "ADMISSION_TYPE",
        "INSURANCE",
        "RELIGION",
        "ETHNICITY",
        "MARITAL_STATUS",
        "GENDER",
    ]
    df_admission = pd.get_dummies(df_admission, prefix=prefix_cols, columns=dummy_cols)

    return df_admission


def preprocess_icu_table_merge_admission(df_admission, df_icu):
    """
    Preprocessing ICU table by converting all to ICU type , having binary features ICU and NICU
    :param df_admission:
    :param df_icu:
    :return:
    """
    # ICU table preprocessing
    df_icu["FIRST_CAREUNIT"].replace(
        {"CCU": "ICU", "CSRU": "ICU", "MICU": "ICU", "SICU": "ICU", "TSICU": "ICU"},
        inplace=True,
    )
    df_icu["cat"] = df_icu["FIRST_CAREUNIT"]
    icu_list = df_icu.groupby("HADM_ID")["cat"].apply(list).reset_index()
    icu_list = df_icu[["HADM_ID", "LOS"]].merge(icu_list, on="HADM_ID")
    # Create admission-ICU matrix
    icu_item = pd.get_dummies(icu_list["cat"].apply(pd.Series).stack()).sum(level=0)
    icu_item[icu_item >= 1] = 1
    icu_item = icu_item.join(icu_list[["HADM_ID", "LOS"]], how="outer")
    # Merge ICU data with main dataFrame
    df_admission = df_admission.merge(icu_item, how="outer", on="HADM_ID")
    # Replace NaNs with 0
    df_admission["ICU"].fillna(value=0, inplace=True)
    df_admission["NICU"].fillna(value=0, inplace=True)

    return df_admission


def preprocess_df_admission(df_admission):
    """
    Preprocessing admission dataframe and constructing meaningful demographics featurs
    :param df_admission: dataframe of admission table of MIMIC III
    :return: processed admission dataframe
    """

    # Remove those that had PU at admission

    indir_kw = [
        "Pressure Ulcer Prevention",
        "Skin Surveillance",
        "decubitus ulcers",
        "decubitus ulcer",
        "Decubitus Ulcers",
        "Decubitus ulcers",
        "Impaired Tissue Integrity",
        "Impaird Skin Integrity",
        "Bedsores",
        "Bed Sore",
        "Bed Sores",
        "Bedsore",
        "decub",
    ]
    indir_kw_low = [i.lower() for i in indir_kw]
    # maybe add coccyx to keywords too?
    kw = [
        "pressure ulcer",
        "Pressure Ulcer",
        " pressure ulcer",
        "Pressure ulcer",
        "pressure Ulcer",
        "pressure ulcers",
        "Pressure Ulcers",
        " pressure ulcers",
        "Pressure ulcers",
        "pressure Ulcers",
        "pressure sore",
        "Pressure sore",
        " pressure Sores",
        "Pressure Sores",
        "pressure sores",
    ]

    PU_ICD_codes = [
        # "70715",
        "70705",
        "70703",
        "70707",
        "70706",
        # "70714",
        "70724",
        # "70719",
        "7070",
        "70721",
        "70722",
        "70720",
        "70711",
        "70723",
        "70710",
        # "70713",
        # "70712",
        "70702",
        "70725",
        "70704",
        "70700",
        "70709",
        "70701",
    ]
    PU_ICD_codes_dotted = [
        # "707.15",
        "707.05",
        "707.03",
        "707.07",
        "707.06",
        # "707.14",
        "707.24",
        # "707.19",
        "707.0",
        "707.21",
        "707.22",
        "707.20",
        "707.11",
        "707.23",
        "707.10",
        # "707.13",
        #         "707.12",
        "707.02",
        "707.25",
        "707.04",
        "707.00",
    ]
    kw += PU_ICD_codes + PU_ICD_codes_dotted + indir_kw + indir_kw_low

    all_diags_pu_mention = np.logical_or.reduce(
        [df_admission["DIAGNOSIS"].str.lower().str.contains(word) for word in kw]
    )
    df_admission["PU_at_admission"] = pd.DataFrame(
        data=all_diags_pu_mention, index=df_admission.index
    )
    df_admission = df_admission[df_admission["PU_at_admission"] != True]
    df_admission.drop(columns=["PU_at_admission"], inplace=True)

    # Convert admission and discharge times to datatime type

    df_admission["ADMITTIME"] = pd.to_datetime(df_admission["ADMITTIME"])
    df_admission["DISCHTIME"] = pd.to_datetime(df_admission["DISCHTIME"])

    # Convert timedelta type into float 'days', 86400 seconds in a day
    df_admission["LOS_total"] = (
        df_admission["DISCHTIME"] - df_admission["ADMITTIME"]
    ).dt.total_seconds() / float(86400)
    # Drop rows with negative LOS, usually related to a time of death before admission
    df_admission = df_admission[df_admission["LOS_total"] > 0]
    # Pre-emptively drop some columns that I don't need anymore
    df_admission.drop(
        columns=[
            "DISCHTIME",
            "ROW_ID",
            "EDREGTIME",
            "EDOUTTIME",
            "HOSPITAL_EXPIRE_FLAG",
            "HAS_CHARTEVENTS_DATA",
        ],
        inplace=True,
    )

    # I don't need to exclude patients who die in the hospital
    df_admission["DECEASED"] = (
        df_admission["DEATHTIME"].notnull().map({True: 1, False: 0})
    )

    # Compress the number of ethnicity categories
    df_admission["ETHNICITY"].replace(regex=r"^ASIAN\D*", value="ASIAN", inplace=True)
    df_admission["ETHNICITY"].replace(regex=r"^WHITE\D*", value="WHITE", inplace=True)
    df_admission["ETHNICITY"].replace(
        regex=r"^HISPANIC\D*", value="HISPANIC/LATINO", inplace=True
    )
    df_admission["ETHNICITY"].replace(
        regex=r"^BLACK\D*", value="BLACK/AFRICAN AMERICAN", inplace=True
    )
    df_admission["ETHNICITY"].replace(
        [
            "UNABLE TO OBTAIN",
            "OTHER",
            "PATIENT DECLINED TO ANSWER",
            "UNKNOWN/NOT SPECIFIED",
        ],
        value="OTHER/UNKNOWN",
        inplace=True,
    )
    df_admission["ETHNICITY"].loc[
        ~df_admission["ETHNICITY"].isin(
            df_admission["ETHNICITY"].value_counts().nlargest(5).index.tolist()
        )
    ] = "OTHER/UNKNOWN"

    # Fix NaNs and file under 'UNKNOWN' for marriage
    df_admission["MARITAL_STATUS"] = df_admission["MARITAL_STATUS"].fillna(
        "UNKNOWN (DEFAULT)"
    )

    return df_admission


def preprocess_df_diagnosis(df_diagnosis):

    """
    Preprocessing diagnosis dataframe to convert ICD9_CODEs to human readable classes of conditions
    :param df_diagnosis: dataframe of diagnosis table of MIMIC III
    :return:
    """

    df_diagnosis["recode"] = df_diagnosis["ICD9_CODE"]
    df_diagnosis["recode"] = df_diagnosis["recode"][
        ~df_diagnosis["recode"].str.contains("[a-zA-Z]").fillna(False)
    ]
    df_diagnosis["recode"].fillna(value="999", inplace=True)
    # https://stackoverflow.com/questions/46168450/replace-specific-range-of-values-in-data-frame-pandas
    df_diagnosis["recode"] = df_diagnosis["recode"].str.slice(start=0, stop=3, step=1)
    df_diagnosis["recode"] = df_diagnosis["recode"].astype(int)

    # ICD-9 Main Category ranges
    icd9_ranges = [
        (1, 140),
        (140, 240),
        (240, 280),
        (280, 290),
        (290, 320),
        (320, 390),
        (390, 460),
        (460, 520),
        (520, 580),
        (580, 630),
        (630, 680),
        (680, 706),
        (706, 707),
        (707, 710),
        (710, 740),
        (740, 760),
        (760, 780),
        (780, 800),
        (800, 998),
        (998, 2000),
    ]

    # Associated category names
    diag_dict = {
        0: "infectious",
        1: "neoplasms",
        2: "endocrine",
        3: "blood",
        4: "mental",
        5: "nervous",
        6: "circulatory",
        7: "respiratory",
        8: "digestive",
        9: "genitourinary",
        10: "pregnancy",
        11: "skin_before_PU",
        12: "Pressure_ulcer",
        13: "skin_after_PU",
        14: "muscular",
        15: "congenital",
        16: "prenatal",
        17: "ill_defined",
        18: "injury",
        19: "external",
    }

    # Re-code in terms of integer
    for num, cat_range in enumerate(icd9_ranges):
        df_diagnosis["recode"] = np.where(
            df_diagnosis["recode"].between(cat_range[0], cat_range[1]),
            num,
            df_diagnosis["recode"],
        )

#     Fixing not PU diagnosis
#     7070 70700 70701 70702 70703 70704 70705 70706 70707 70709 7071 70710 70711 70712 70713 70714 70715 70719 70720 70721
#     70722 70723 70724 70725 7078 7079
    black_PU_list = ["7078","70708", "70712", "70713", "70714", "70715", "70719"]
    not_pu_indexes = df_diagnosis["ICD9_CODE"].isin(black_PU_list)
    df_diagnosis.loc[not_pu_indexes, "recode"] = 13

    # Convert integer to category name using diag_dict
    df_diagnosis["recode"] = df_diagnosis["recode"]
    df_diagnosis["cat"] = df_diagnosis["recode"].replace(diag_dict)

    return df_diagnosis


def preprocess_patient_df_and_merge_into_admission(df_patient, df_admission):
    """
    Preprocessing patient table and merginig into addmission
    :param df_patient: dataframe of patient table of MIMIC III
    :param df_admission: dataframe of admission table of MIMIC III
    :return: processed and merged addmission dataframe
    """
    # Convert to datetime type
    df_patient["DOB"] = pd.to_datetime(df_patient["DOB"])
    df_patient = df_patient[["SUBJECT_ID", "DOB", "GENDER"]]

    # Adding patient information to addmission table
    # age
    df_admission = df_admission.merge(df_patient, how="inner", on="SUBJECT_ID")
    # Find the first admission time for each patient
    df_age_min = (
        df_admission[["SUBJECT_ID", "ADMITTIME"]]
        .groupby("SUBJECT_ID")
        .min()
        .reset_index()
    )
    df_age_min.columns = ["SUBJECT_ID", "ADMIT_MIN"]

    df_admission = df_admission.merge(df_age_min, how="outer", on="SUBJECT_ID")
    df_admission["DOB"] = pd.to_datetime(df_admission["DOB"]).dt.date
    df_admission["ADMIT_MIN"] = pd.to_datetime(df_admission["ADMIT_MIN"]).dt.date
    df_admission["age"] = df_admission.apply(
        lambda e: ((e["ADMIT_MIN"] - e["DOB"]).days / 365.25), axis=1
    )
    # People who have age <0 are most likely emergency cases who were already dead, set them to default of 90
    #     df_admission["age"] = np.where(df_admission["age"] < 0, 90, df_admission["age"])
    df_admission["age"] = np.where(df_admission["age"] >= 300, 90, df_admission["age"])
    # Removing kids
    return df_admission


def create_stage_notes_PU_Flag(
    df_admission, df_patient, df_diagnosis, df_icu, df_note, staging_df, model_input_path
):
    """
    Creates a single feature matrix from the tables in MIMIC III data
    :param df_admission: dataframe of addmission table of MIMIC III
    :param df_patient: dataframe of patient table of MIMIC III
    :param df_diagnosis: dataframe of diagnosis table of MIM
    :param df_icu: dataframe of icu table of MIMIC III
    :param df_note: dataframe of note table of MIMIC III
    :return: Single dataframe (i.e feature matrix) containing demographics and count values of different dianostic classes
    """

    stage_ids = return_pos_stage_data(df_admission, df_patient, staging_df)

    note_pu_ids = marking_PU_in_notes(df_note, df_admission)

    # Preprocessing the admission data

    df_admission = preprocess_df_admission(df_admission)

    # Preprocessing diagnois table

    df_diagnosis = preprocess_df_diagnosis(df_diagnosis)

    # create a dummy matrix that highlights all the diagnoses for each admission
    hadm_list = df_diagnosis.groupby("HADM_ID")["cat"].apply(list).reset_index()
    # Convert diagnoses list into hospital admission-item matrix
    hadm_item = pd.get_dummies(hadm_list["cat"].apply(pd.Series).stack()).sum(level=0)
    hadm_item.head()
    # Join back with HADM_ID, will merge with main admissions DF later
    hadm_item = hadm_item.join(hadm_list["HADM_ID"], how="outer")
    hadm_item.head()
    # Merge diagnosis information with main admissions df
    df_admission = df_admission.merge(hadm_item, how="inner", on="HADM_ID")

    # Preprocessing patients table

    df_admission = preprocess_patient_df_and_merge_into_admission(
        df_patient, df_admission
    )

    # Preprocessing Icu table and merging it into admission table

    df_admission = preprocess_icu_table_merge_admission(df_admission, df_icu)

    # Final cleaning on the admission table

    df_admission = final_admission_cleaning(df_admission)

    # Finding PU mention in notes

    # groupby df_note by hadm_id

    #     new_df_admission = df_admission.merge(
    #         df_PU_note, how="outer", on="HADM_ID", indicator=True
    #     )
    #     clean_df_admission = new_df_admission[new_df_admission["_merge"] == "both"]
    #     clean_df_admission = df_admission[new_df_admission["_merge"] == "both"]

    #     clean_df_admission.drop(columns=["_merge"], inplace=True)

    df_admission["Stage_PU_positive"] = df_admission["HADM_ID"].isin(stage_ids)
    df_admission["note_PU_positive"] = df_admission["HADM_ID"].isin(note_pu_ids)

    df_admission["PU_mention_in_both"] = (
        (df_admission["Stage_PU_positive"] > 0) | (df_admission["note_PU_positive"] > 0)
    ).astype(int)

#     df_admission = df_admission[
#         (df_admission["PU_mention_in_both"] != 0)
#         | (
#             (df_admission["PU_mention_in_both"] == 0)
#             & (df_admission["Pressure_ulcer"] < 1)
#         )
#     ]
    # Adding icu losses

    los_summed_clean_df_admission = df_admission.groupby("HADM_ID")["LOS"].transform(
        sum
    )
    df_admission["LOS"] = los_summed_clean_df_admission
    df_admission.drop_duplicates(inplace=True)
    # writing to csv
    df_admission.to_csv(model_input_path + "/jamia_note_stage_pu_marked.csv", index=False)

    return df_admission


def marking_PU_in_notes(df_note):
    """
    Using keywords and  to mark pressure ulcers in each admission
    :param df_note: dataframe of the note table of MIMIC III
    :return: PU marked dataframe
    """
    
#     bed sore, bed ulcer, pressure sore, pressure ulcer, decub* sore, decub* ulcer

#     patterns = bedsore($|[^:]) decub\w* sore($|[^:])
#     (\w*\s*) sore($|[^:])
    patterns =['(bed sore($|[^:]))', '(bed ulcer($|[^:]))', '(pressure sore($|[^:]))', '(pressure ulcer($|[^:]))', '(decub(\w*\s*) sore($|[^:]))', '(decub(\w*\s*) ulcer($|[^:]))']
    df_note = df_note[["HADM_ID", "TEXT","CHARTDATE",'CHARTTIME']]
    df_note["TEXT"] = df_note["TEXT"].str.lower()
    all_notes_pu_mention = np.logical_or.reduce(
        [df_note["TEXT"].str.contains(word, regex = True) for word in patterns]
    )
    df_note["PU_mention_notes"] = pd.DataFrame(
        data=all_notes_pu_mention, index=df_note.index
    )
#     df_note.drop(columns=["TEXT"], inplace=True)

    return df_note

def extract_earliest_date(data_path, model_input_path, kind):

    df_admission = pd.read_csv(data_path +'/ADMISSIONS.csv')
    df_patient = pd.read_csv(data_path +'/PATIENTS.csv')
    staging_df = pd.read_csv(Pu_marked_charts + "PU_chart_vivian_ids.csv")
    # Adding patient information to ddmission table
    df_patient = df_patient[["SUBJECT_ID", "DOB", "GENDER"]]
    df_admission = df_admission.merge(df_patient, how = 'inner', on="SUBJECT_ID")

    df_age_min = (
    df_admission[["SUBJECT_ID", "ADMITTIME"]]
    .groupby("SUBJECT_ID")
    .min()
    .reset_index()
    )
    df_age_min.columns = ["SUBJECT_ID", "ADMIT_MIN"]
    df_admission = df_admission.merge(df_age_min, how="outer", on="SUBJECT_ID")
    df_admission['DOB'] = pd.to_datetime(df_admission['DOB']).dt.date
    df_admission['ADMIT_MIN'] = pd.to_datetime(df_admission['ADMIT_MIN']).dt.date
    df_admission["age"] = df_admission.apply(
    lambda e: ((e["ADMIT_MIN"] - e["DOB"]).days / 365.25), axis=1
    )

    # Removing kids
    df_admission = df_admission[df_admission["age"] >= min_age]


    df_admission_alive = df_admission[df_admission['DEATHTIME'].isna()]

    df_admission_alive = df_admission[df_admission['DEATHTIME'].isna()]


    if not consider_dead:
        df_admission = df_admission[df_admission['DEATHTIME'].isna()]

    df_adm_time = df_admission[['ADMITTIME', 'HADM_ID']]
    # care_items = []
    # meta_items = []
    if egem:
        meta_items = [224631, 224965, 224966]
    else:
        meta_items = [224631,224965,224966,224967,224968,224969,224970,224971,227618,227619]

    care_items = [551,552,553]
    care_data = staging_df[staging_df['itemid'].isin(care_items)]
    meta_data = staging_df[staging_df['itemid'].isin(meta_items)]
    care_vals = list(care_data['value'].unique())
    meta_vals = list(meta_data['value'].unique())


    if ours or egem:
        adm_not_case_vals = ['Other/Remarks','Not applicable', "None", "Negative", "No", 'Not Applicable','Unable to Stage','Unable to assess; dressing not removed','Unable to stage; wound is covered with eschar']
    else:
        adm_not_case_vals = ['Other/Remarks','Not applicable', "None", "Negative", "No", 'Not Applicable']

    disch_not_case_vals = ['Other/Remarks','Not applicable', "None", "Negative", "No", 'Not Applicable']
    admission_case_vals = [i for i in care_vals+meta_vals  if i not in adm_not_case_vals]
    discharge_case_vals = [i for i in care_vals+meta_vals  if i not in disch_not_case_vals]
    all_case_vals = [i for i in care_vals+meta_vals]


    staging_df = staging_df.dropna(subset = ['value'])
    staging_df = staging_df.dropna(subset = ['charttime'])
    staging_df = staging_df.sort_values(by = 'charttime')

    # the_very_first_date = staging_df.groupby("hadm_id")[["charttime",'value']].first().reset_index()
    pui_staging_df = staging_df[staging_df['value'].isin(all_case_vals)]

    the_very_first_date_after24 = df_adm_time.merge(pui_staging_df,left_on = "HADM_ID" , right_on = "hadm_id", how = 'inner')
    the_very_first_date_after24["charttime"] = pd.to_datetime(the_very_first_date_after24["charttime"])
    the_very_first_date_after24["ADMITTIME"] = pd.to_datetime(the_very_first_date_after24["ADMITTIME"])
    the_very_first_date_after24['time_to_first_stage'] = the_very_first_date_after24.apply(
    lambda e: ((e["charttime"] - e["ADMITTIME"]).total_seconds()/float(3600)), axis=1
    )


    the_very_first_date_after24 = the_very_first_date_after24[the_very_first_date_after24["time_to_first_stage"] > 24]
    the_very_first_date_after24 = the_very_first_date_after24.sort_values(by = 'charttime')
    the_very_first_date_after24 = the_very_first_date_after24.groupby("hadm_id")[["charttime",'value']].first().reset_index()


    adm_staging_df = staging_df[staging_df['value'].isin(admission_case_vals)]
    adm_staging_df = adm_staging_df.sort_values(by = 'charttime')

    disch_staging_df = staging_df[staging_df['value'].isin(discharge_case_vals)]
    disch_staging_df = disch_staging_df.sort_values(by = 'charttime')

    if ours:
        staging_dict_start = { 'Intact,Color Chg':1 ,'Red; unbroken':1, 'Red, Unbroken':1
        , 'Through Dermis':2, 'Through Fascia':3, 'To Bone':4,'Part. Thickness':2, 'Full Thickness':3,
                        'Partial thickness skin loss through epidermis and/or dermis; ulcer may present as an abrasion, blister, or shallow crater':2,
                        'Full thickness skin loss that may extend down to underlying fascia; ulcer may have tunneling or undermining':3,
                        'Full thickness skin loss with damage to muscle, bone, or supporting structures; tunneling or undermining may be present':4,
                        'Deep tissue injury':4,
                         "Deep Tiss Injury":4,
        }

        staging_dict_end = { 'Intact,Color Chg':1 ,'Red; unbroken':1, 'Red, Unbroken':1
        , 'Through Dermis':2, 'Through Fascia':3, 'To Bone':4,'Part. Thickness':2, 'Full Thickness':3,
                        'Partial thickness skin loss through epidermis and/or dermis; ulcer may present as an abrasion, blister, or shallow crater':2,
                        'Full thickness skin loss that may extend down to underlying fascia; ulcer may have tunneling or undermining':3,
                        'Full thickness skin loss with damage to muscle, bone, or supporting structures; tunneling or undermining may be present':4,
                        'Deep tissue injury':3,
                         "Deep Tiss Injury":3,
                        'Unable to assess; dressing not removed':5,'Unable to stage; wound is covered with eschar':5 ,'Unable to Stage':5,
        }
    elif egem:
        staging_dict_start = { 'Intact,Color Chg':1 ,'Red; unbroken':1, 'Red, Unbroken':1
        , 'Through Dermis':2, 'Through Fascia':3, 'To Bone':4,'Part. Thickness':2, 'Full Thickness':3,
                        'Partial thickness skin loss through epidermis and/or dermis; ulcer may present as an abrasion, blister, or shallow crater':2,
                        'Full thickness skin loss that may extend down to underlying fascia; ulcer may have tunneling or undermining':3,
                        'Full thickness skin loss with damage to muscle, bone, or supporting structures; tunneling or undermining may be present':4,
                        'Deep tissue injury':0,
                         "Deep Tiss Injury":0,
        }
    else:
        staging_dict_start = { 'Intact,Color Chg':1 ,'Red; unbroken':1, 'Red, Unbroken':1
        , 'Through Dermis':2, 'Through Fascia':3, 'To Bone':4,'Part. Thickness':2, 'Full Thickness':3,
                        'Partial thickness skin loss through epidermis and/or dermis; ulcer may present as an abrasion, blister, or shallow crater':2,
                        'Full thickness skin loss that may extend down to underlying fascia; ulcer may have tunneling or undermining':3,
                        'Full thickness skin loss with damage to muscle, bone, or supporting structures; tunneling or undermining may be present':4,
                        'Deep tissue injury':3,
                         "Deep Tiss Injury":3,
                        'Unable to assess; dressing not removed':5,'Unable to stage; wound is covered with eschar':5 ,'Unable to Stage':5,}



    first_df = adm_staging_df.groupby("hadm_id")[["charttime",'value']].first().reset_index()

    first_df['first_stage'] = first_df['value'].map(staging_dict_start)
    first_df.columns = ["hadm_id","first_charttime",'first_value','first_stage']
    if ours:
        last_df = disch_staging_df.groupby("hadm_id")[["charttime",'value']].last().reset_index()
        last_df['last_stage'] = last_df['value'].map(staging_dict_end)
        last_df.columns = ["hadm_id","last_charttime",'last_value','last_stage']

    case_df_adm = df_adm_time.merge(first_df,left_on = "HADM_ID" , right_on = "hadm_id", how = 'inner')
    case_df_adm.drop(columns = ['HADM_ID'], inplace = True)
    case_df_adm["first_charttime"] = pd.to_datetime(case_df_adm["first_charttime"])
    case_df_adm["ADMITTIME"] = pd.to_datetime(case_df_adm["ADMITTIME"])
    case_df_adm['time_to_first_stage'] = case_df_adm.apply(
    lambda e: ((e["first_charttime"] - e["ADMITTIME"]).total_seconds()/float(3600)), axis=1
    )


    if ours:
        case_df_real_admission_within_oneday = case_df_adm[case_df_adm["time_to_first_stage"] <= cut_off_hour]
        case_df_disch = df_adm_time.merge(last_df,left_on = "HADM_ID" , right_on = "hadm_id", how = 'inner')
        case_df_disch.drop(columns = ['HADM_ID'], inplace = True)
        case_df_disch["last_charttime"] = pd.to_datetime(case_df_disch["last_charttime"])
        case_df_disch["ADMITTIME"] = pd.to_datetime(case_df_disch["ADMITTIME"])
        case_df_disch['time_to_last_stage'] = case_df_disch.apply(
        lambda e: ((e["last_charttime"] - e["ADMITTIME"]).total_seconds()/float(3600)), axis=1
        )
        case_df_real_discharge_after_oneday = case_df_disch[case_df_disch["time_to_last_stage"] > cut_off_hour]

        case_df_real_discharge_after_oneday_greater_than_two = case_df_real_discharge_after_oneday[case_df_real_discharge_after_oneday['last_stage'] >= min_stage]
        final_df  = case_df_real_discharge_after_oneday_greater_than_two.merge(case_df_real_admission_within_oneday, on = 'hadm_id', how = 'left')
        final_df['first_stage'] = final_df['first_stage'].fillna(0)
        final_df_worsened = final_df[final_df['last_stage'] > final_df['first_stage'] ]
        final_df_worsened_lim = final_df_worsened[['hadm_id','last_value','last_stage','time_to_last_stage','first_value','first_stage','time_to_first_stage']]
        final_df_worsened_lim.head(5)
        pos_hams = final_df_worsened_lim['hadm_id']
    else:
        case_df_real_admission_within_oneday = case_df_adm[case_df_adm["time_to_first_stage"] > cut_off_hour]
        final_df_worsened_lim = case_df_real_admission_within_oneday[case_df_real_admission_within_oneday["first_stage"] >= min_stage ]
        pos_hams = final_df_worsened_lim['hadm_id']


    the_very_first_date_after24.shape

    the_very_first_date_after24

    the_very_first_date_after24.to_csv(model_input_path + 'the_very_first_date_after24_stage_' + kind + '.csv',index = False)
    
    


    df_note = pd.read_csv(data_path +'/NOTEEVENTS.csv')


    marked_note = marking_PU_in_notes(df_note)

    pos_note = marked_note[marked_note['PU_mention_notes'] > 0]
    pos_note['CHARTTIME'] = pd.to_datetime(pos_note['CHARTTIME'])
    pos_note['CHARTDATE'] = pd.to_datetime(pos_note['CHARTDATE'])
    pos_note['CHARTTIME'] = pos_note['CHARTTIME'].fillna(pos_note['CHARTDATE']-pd.Timedelta(days=0))
    pos_note_by_hadm= (
        pos_note.groupby(["HADM_ID"])["CHARTDATE","CHARTTIME"].min().reset_index()

    )

    pos_note_by_hadm = pos_note_by_hadm.merge(df_adm_time, on = 'HADM_ID')

    pos_note_by_hadm['ADMITTIME'] = pd.to_datetime(pos_note_by_hadm['ADMITTIME'])
    pos_note_by_hadm['CHARTDATE'] = pd.to_datetime(pos_note_by_hadm['CHARTDATE'])
    pos_note_by_hadm['CHARTTIME'] = pd.to_datetime(pos_note_by_hadm['CHARTTIME'])

    pos_note_by_hadm['time_to_first_keyw_app'] = pos_note_by_hadm.apply(
        lambda e: ((e["CHARTDATE"] - e["ADMITTIME"]).total_seconds()/float(3600)), axis=1
        )
    pos_note_by_hadm['time_to_first_keyw_app_acc'] = pos_note_by_hadm.apply(
        lambda e: ((e["CHARTTIME"] - e["ADMITTIME"]).total_seconds()/float(3600)), axis=1
        )
    # pos_note_by_hadm['time_to_first_keyw_app_acc_day'] = pos_note_by_hadm.apply(
    #     lambda e: ((e["CHARTTIME"] - e["ADMITTIME"]).total_seconds()/float(24 * 3600)), axis=1
    #     )
    pos_note_by_hadm_after48 = pos_note_by_hadm[((pos_note_by_hadm['time_to_first_keyw_app_acc']> cut_off_hour))]

    pos_note_by_hadm_after48.head()
    cols =[ 'HADM_ID','CHARTTIME','ADMITTIME','time_to_first_keyw_app_acc']
    pos_note_by_hadm_after48 = pos_note_by_hadm_after48[cols]
    pos_note_by_hadm_after48.to_csv(model_input_path, 'notes_dates_'+ kind+ '.csv',index = False)


def add_amia_cohort(model_input_path):
    
    data = pd.read_csv(model_input_path + "jamia_note_stage_pu_marked.csv")

    egem_data = pd.read_csv(model_input_path + "/limnote_only_note_threepapers_egem.csv")

    data_merged = data.merge(egem_data, on = 'HADM_ID')


    data_merged['PU_mention_in_both'] = data_merged['note_PU_positive'] & data_merged['Pressure_ulcer'] > 0

    (data_merged['PU_mention_in_both'] == True).sum()

    data_lim = data_merged[[ 'HADM_ID', 'TEXT','ground_truth','PU_mention_in_both']]

    data_lim = data_lim.sort_values(by = 'HADM_ID')

    data_lim['PU_mention_in_both'] = data_lim['PU_mention_in_both'].astype(int)

    data_lim.to_csv(model_input_path + "/limnote_only_note_threepapers_aimia.csv", index = False)


def define_cohorts(data_path, model_input_path):

    for kind in ['ours', 'Jamia', 'egem']:
        
        if kind == 'ours': 
            min_age = 15
            cut_off_hour = 24
            consider_dead = False
            min_stage = 2
            egem = False
            ours = True

        if kind == 'Jamia' :
            min_age = 15
            cut_off_hour = 48
            consider_dead = True
            min_stage = 1
            egem = False
            ours = False

        if kind == 'egem':
            min_age = 18
            cut_off_hour = 24
            consider_dead = True
            min_stage = 2
            egem = True
            ours = False

        df_note = pd.read_csv(data_path +'/NOTEEVENTS.csv')
        df_admission = pd.read_csv(data_path +'/ADMISSIONS.csv')
        df_patient = pd.read_csv(data_path +'/PATIENTS.csv')
        df_patient = pd.read_csv(data_path +'/PATIENTS.csv')

        # Diagnosis for each admission to hospital
        df_diagnosis = pd.read_csv(data_path +'/DIAGNOSES_ICD.csv')

        # Intensive Care Unit (ICU) for each admission to hospital
        df_icu = pd.read_csv(data_path +'/ICUSTAYS.csv')
        df_cpt = pd.read_csv(data_path +'/CPTEVENTS.csv')
        
        df_chart = pd.read_csv(data_path +'/CHARTEVENTS.csv')
        
        
        PU_item_ids = [83,84,85,86,87,88,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,8457,8458,8459,8460,8461,8462,8463,8464,8465,224631,224965,224966,224967,224968,224969,224970,224971,227618,227619,228506,228507,228508,228509,228510,228511,228512,228513,228514,228515,228539,228540,228541,228542,228543,228544,228545,228546,228547,228548,228549,228550,228551,228552,228553,228554,228555,228556,228557,228558,228610,228611,228612,228613,228614,228615,228616,228617,228618,228619,228620,228621,228622,228623,228624,228625,228626,228627,228628,228629]
        
        staging_df = df_chart[df_chart['itemid'].isin(PU_item_ids)]

        
        staging_df.to_csv(model_input_path + "PU_chart_vivian_ids.csv", index = False)
        # extracting pu related staging table from mimic iii table 
        
#         CREATE TABLE PU_chart_vivian_ids AS
#   select * from mimiciii.chartevents where itemid in (83,84,85,86,87,88,551,552,553,554,555,556,557,558,559,560,561,562,563,564,565,566,567,568,569,570,571,572,573,574,575,576,577,578,579,8457,8458,8459,8460,8461,8462,8463,8464,8465,224631,224965,224966,224967,224968,224969,224970,224971,227618,227619,228506,228507,228508,228509,228510,228511,228512,228513,228514,228515,228539,228540,228541,228542,228543,228544,228545,228546,228547,228548,228549,228550,228551,228552,228553,228554,228555,228556,228557,228558,228610,228611,228612,228613,228614,228615,228616,228617,228618,228619,228620,228621,228622,228623,228624,228625,228626,228627,228628,228629);

# \copy PU_chart_vivian_ids to /data/mimic/PU_chart_vivian_ids.csv csv header



        staging_df = pd.read_csv(Pu_marked_charts + "PU_chart_vivian_ids.csv")
        
        
        # Adding patient information to ddmission table
        df_patient = df_patient[["SUBJECT_ID", "DOB", "GENDER"]]
        df_admission = df_admission.merge(df_patient, how = 'inner', on="SUBJECT_ID")
        # Find the first admission time for each patient
        df_age_min = (
        df_admission[["SUBJECT_ID", "ADMITTIME"]]
        .groupby("SUBJECT_ID")
        .min()
        .reset_index()
        )
        df_age_min.columns = ["SUBJECT_ID", "ADMIT_MIN"]
        df_admission = df_admission.merge(df_age_min, how="outer", on="SUBJECT_ID")
        df_admission['DOB'] = pd.to_datetime(df_admission['DOB']).dt.date
        df_admission['ADMIT_MIN'] = pd.to_datetime(df_admission['ADMIT_MIN']).dt.date
        df_admission["age"] = df_admission.apply(
        lambda e: ((e["ADMIT_MIN"] - e["DOB"]).days / 365.25), axis=1
        )

        # Removing kids
        df_admission = df_admission[df_admission["age"] > min_age]
        # only keeping alive ones 
        if not consider_dead:
            df_admission = df_admission[df_admission['DEATHTIME'].isna()]
        admiss_ids = df_admission['HADM_ID'].values
        real_df_note = df_note[df_note['HADM_ID'].isin(admiss_ids)]

        len(df_note['HADM_ID'].unique())

        len(real_df_note['HADM_ID'].unique())

        extract_earliest_date(data_path, model_input_path , kind)
        
        
        create_stage_notes_PU_Flag(
    df_admission, df_patient, df_diagnosis, df_icu, df_note, staging_df, model_input_path
)
        
        cohort_data = model_input_path + 'the_very_first_date_after24_stage_' + kind + '.csv'
        stage = pd.read_csv(cohort_data)
        note = pd.read_csv(model_input_path, 'notes_dates_'+ kind+ '.csv')
        note.rename(columns = {"CHARTTIME":"notetime"},inplace = True)
        stage.rename(columns = {"charttime":"stagetime"},inplace = True)

        real_df_note = real_df_note.merge(stage,left_on = ['HADM_ID'], right_on = ['hadm_id'], how = 'left')
        real_df_note.drop(columns = ['hadm_id'], inplace = True)

        real_df_note = real_df_note.merge(note,on = ['HADM_ID'],how = 'left')

        real_df_note['notetime'] = pd.to_datetime(real_df_note['notetime'])
        real_df_note['stagetime'] = pd.to_datetime(real_df_note['stagetime'])
        real_df_note['CHARTTIME'] = pd.to_datetime (real_df_note['CHARTTIME'])
        real_df_note['min_time'] = real_df_note[['notetime','stagetime']].min(axis = 1)

        real_df_note['notetime'].isna().sum()

        real_df_note["CHARTTIME"].fillna(pd.to_datetime(real_df_note["CHARTDATE"]) - pd.Timedelta(nanoseconds=0), inplace=True)


        len(real_df_note[real_df_note['min_time'].isna()]['HADM_ID'].unique())

        pos_hadms = (real_df_note[~real_df_note['min_time'].isna()]['HADM_ID'].unique())

        filtered_df_note_lim_grouped

        real_df_note

        df_note_pos = real_df_note[real_df_note['HADM_ID'].isin(pos_hadms)]

        df_note_neg = real_df_note[~real_df_note['HADM_ID'].isin(pos_hadms)]

        df_note_pos_sort = df_note_pos.sort_values(by = ['HADM_ID','CHARTDATE','CHARTTIME'])
        df_note_neg_sort = df_note_neg.sort_values(by = ['HADM_ID','CHARTDATE','CHARTTIME'])



        df_note_pos_sort_grr = df_note_pos_sort[['HADM_ID','CHARTTIME','min_time']].groupby('HADM_ID').first().reset_index()
        df_note_neg_sort_gr = df_note_neg_sort[['HADM_ID','CHARTTIME','min_time']].groupby('HADM_ID').first().reset_index()

        df_note_pos_sort_grr['note_duration'] = df_note_pos_sort_grr['min_time'] - df_note_pos_sort_grr['CHARTTIME']

        df_note_pos_sort_grr['note_duration'].min()

        df_note_pos_sort_grr['note_duration'] = df_note_pos_sort_grr['note_duration'].apply(lambda x: x.total_seconds() / float(3600*24))

        df_note_pos_sort_grr.drop(columns = ['CHARTTIME'], inplace = True)


        df_note_pos_sort_grr

        df_note_pos_sort_gr = df_note_pos_sort_grr[df_note_pos_sort_grr['note_duration'] > 0]

        # Set up empty lists to stroe results
        chi_square = []
        p_values = []

        size = df_note_pos_sort_gr['note_duration'].shape[0]

        # Set up 50 bins for chi-square test
        # Observed data will be approximately evenly distrubuted aross all bins
        percentile_bins = np.linspace(0,100,51)
        percentile_cutoffs = np.percentile(df_note_pos_sort_gr['note_duration'], percentile_bins)
        observed_frequency, bins = (np.histogram(df_note_pos_sort_gr['note_duration'], bins=percentile_cutoffs))
        cum_observed_frequency = np.cumsum(observed_frequency)
        # 'johnsonsb','johnsonsu',
        list_of_dists = ['alpha','anglit','arcsine','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','dgamma','dweibull','erlang','expon','exponnorm','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','genpareto','gennorm','genexpon','genextreme','gausshyper','gamma','gengamma','genhalflogistic','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss','invweibull','kstwobign','laplace','levy','levy_l','logistic','loggamma','loglaplace','lognorm','lomax','maxwell','mielke','nakagami','ncx2','ncf','nct','norm','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','reciprocal','rayleigh','rice','recipinvgauss','semicircular','t','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','weibull_min','weibull_max']
        for distribution in list_of_dists:
            # Set up distribution and get fitted distribution parameters
            dist = getattr(scipy.stats, distribution)
            param = dist.fit(df_note_pos_sort_gr['note_duration'])

            # Obtain the KS test P statistic, round it to 5 decimal places
            p = scipy.stats.kstest(df_note_pos_sort_gr['note_duration'], distribution, args=param)[1]
            p = np.around(p, 5)
            p_values.append(p)    

            # Get expected counts in percentile bins
            # This is based on a 'cumulative distrubution function' (cdf)
            cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                                  scale=param[-1])
            expected_frequency = []
            for bin in range(len(percentile_bins)-1):
                expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
                expected_frequency.append(expected_cdf_area)

            # calculate chi-squared
            expected_frequency = np.array(expected_frequency) * size
            cum_expected_frequency = np.cumsum(expected_frequency)
            ss = sum (((cum_expected_frequency - cum_observed_frequency) ** 2) / cum_observed_frequency)
            chi_square.append(ss)

        # Collate results and sort by goodness of fit (best at top)

        results = pd.DataFrame()
        results['Distribution'] = list_of_dists
        results['chi_square'] = chi_square
        results['p_value'] = p_values
        results.sort_values(['chi_square'], inplace=True)

        # Report results

        print ('\nDistributions sorted by goodness of fit:')
        print ('----------------------------------------')
        print (results)


        top_dist =  getattr(scipy.stats, results.iloc[0]['Distribution'])


        x = np.arange(size)

        # Divide the observed data into 100 bins for plotting (this can be changed)
        number_of_bins = 100
        bin_cutoffs = np.linspace(np.percentile(df_note_pos_sort_gr['note_duration'],0), np.percentile(df_note_pos_sort_gr['note_duration'],99),number_of_bins)

        # Create the plot
        h = plt.hist(df_note_pos_sort_gr['note_duration'], bins = bin_cutoffs, color='0.75')

        # Get the top three distributions from the previous phase
        number_distributions_to_plot = 1
        dist_names = results['Distribution'].iloc[0:number_distributions_to_plot]

        # Create an empty list to stroe fitted distribution parameters
        parameters = []

        # Loop through the distributions ot get line fit and paraemters

        for dist_name in dist_names:
            # Set up distribution and store distribution paraemters
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(df_note_pos_sort_gr['note_duration'])
            parameters.append(param)

            # Get line for each distribution (and scale to match observed data)
            pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1])
            scale_pdf = np.trapz (h[0], h[1][:-1]) / np.trapz (pdf_fitted, x)
            pdf_fitted *= scale_pdf

            # Add the line to the plot
            plt.plot(pdf_fitted, label=dist_name)

            # Set the plot x axis to contain 99% of the data
            # This can be removed, but sometimes outlier data makes the plot less clear
            plt.xlim(0,np.percentile(df_note_pos_sort_gr['note_duration'],99))

        # Add legend and display plot

        plt.legend()
        plt.show()

        # Store distribution paraemters in a dataframe (this could also be saved)
        dist_parameters = pd.DataFrame()
        dist_parameters['Distribution'] = (
                results['Distribution'].iloc[0:number_distributions_to_plot])
        dist_parameters['Distribution parameters'] = parameters

        # Print parameter results
        print ('\nDistribution parameters:')
        print ('------------------------')

        for index, row in dist_parameters.iterrows():
            print ('\nDistribution:', row[0])
            print ('Parameters:', row[1] )



        num_negs = len(set(real_df_note['HADM_ID'].unique())) - real_df_note[~real_df_note['min_time'].isna()]['HADM_ID'].unique().shape[0]

        num_notes_negs = top_dist.rvs( *param, size=num_negs)

        while np.any(num_notes_negs < [0] * num_negs):
            num_notes_negs = top_dist.rvs( *param, size=num_negs)


        df_note_neg_count_gr = df_note_neg_sort[['HADM_ID','CHARTTIME']].groupby('HADM_ID').count().reset_index()

        df_note_neg_count_gr.columns = ['HADM_ID', 'num_notes']

        df_note_neg_sort_gr = df_note_neg_sort_gr.merge(df_note_neg_count_gr, on = 'HADM_ID')

        df_note_neg_sort_gr.sort_values( by = 'num_notes', inplace = True)

        num_notes_negs = sorted(num_notes_negs) 

        df_note_neg_sort_gr['note_duration'] = num_notes_negs

        df_note_neg_sort_gr['min_time'] = df_note_neg_sort_gr.apply(lambda row: row['CHARTTIME'] + pd.Timedelta(days= max (1, row['note_duration'])), axis = 1)

        df_note_neg_sort_gr_lim = df_note_neg_sort_gr[['HADM_ID','min_time']]

        df_note_pos_sort_grr_lim = df_note_pos_sort_grr[['HADM_ID','min_time']]

        df_mins = pd.concat([df_note_pos_sort_grr_lim,df_note_neg_sort_gr_lim])


        len(real_df_note['HADM_ID'].unique())

        real_df_note.drop(columns = ['min_time'], inplace = True)

        real_df_note = real_df_note.merge(df_mins, on = 'HADM_ID')


        large = ( set(real_df_note['HADM_ID'].unique()))

        # only keeping text before
        filtered_df_note = real_df_note[(real_df_note["CHARTTIME"] < real_df_note['min_time'])]

        filtered_df_note.shape

        small = set(filtered_df_note['HADM_ID'].unique())

        diffak = large.difference(small)

        real_df_note[real_df_note['HADM_ID'].isin(diffak)].sort_values(by = 'HADM_ID')["HADM_ID"].unique()

        df_note_1 = real_df_note[real_df_note['HADM_ID'] == 153510]

        df_note_1_adm = df_note_1.merge(df_admission, on= 'HADM_ID')

        balag= df_note_1_adm[['CHARTDATE','CHARTTIME','TEXT','min_time']].sort_values(by = ['CHARTDATE','CHARTTIME'])

        filtered_df_note.sort_values(by = ['HADM_ID','CHARTTIME'], inplace = True)


        filtered_df_note_pos_lim = filtered_df_note[['HADM_ID', 'min_time']]

        filtered_df_note_lim = filtered_df_note[['HADM_ID','TEXT']]

        filtered_df_note_lim_grouped = filtered_df_note_lim.groupby(['HADM_ID'])['TEXT'].apply(lambda x: "%s" % '\n\n\n'.join(x)).reset_index()



        df_admission = pd.read_csv(data_path +'/ADMISSIONS.csv')

        filtered_df_note_lim_grouped = filtered_df_note_lim_grouped.merge(real_cohort[['ground_truth', 'PU_mention_in_both', 'HADM_ID']], on = ['HADM_ID'],how = 'inner')


        filtered_df_note_lim_grouped.to_csv(model_input_path + "limnote_only_note_threepapers_" + kind +  ".csv", index = False)

    add_amia_cohort(model_input_path)
    

data_path = '/data/mimic/CSVs'

model_input_path = '/data/mimic/PUI_data/data_test/'

# first_date_path = '/data/mimic/PUI_data/data_test/'

# notes_date_path = '/data/mimic/PUI_data/data_test/'

# note_stage_path = "/data/mimic/PUI_data/data_test/"


# Pu_marked_charts = "/data/mimic/PUI_data/data_test/"


define_cohorts(data_path, model_input_path)