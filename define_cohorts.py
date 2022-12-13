import os
os.chdir("/home/mani/nursing/PU_nursing")


import scipy
import numpy as np
import matplotlib.pyplot as plt

from create_note_diagnosis_pu import create_diagnosis_notes_PU_Flag
from create_note_diagnosis_dummy import create_diagnosis_notes_PU_Flag_dummy
from create_note_diagnosis_pu_withcpt import create_diagnosis_notes_PU_Flag_cpt
from create_note_diagnosis_pu_withcpt_in_notes_too import create_diagnosis_notes_PU_Flag_cpt_in_notes_too
from case_control_demographics_real import case_control_demographics
from create_stage_note_pu import create_stage_notes_PU_Flag
from join_diagnosis_labs import join_diagnosis_labs
import pandas as pd
from sklearn.metrics import roc_curve,precision_score,recall_score
from sklearn.metrics import roc_auc_score


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

def extract_earliest_date(data_path, first_date_path, Pu_marked_charts, notes_dates_path, kind)

    df_admission = pd.read_csv(data_path +'/ADMISSIONS.csv')
    df_patient = pd.read_csv(data_path +'/PATIENTS.csv')
    staging_df = pd.read_csv(Pu_marked_charts)
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

    the_very_first_date_after24.to_csv(first_date_path + 'the_very_first_date_after24_stage_' + kind + '.csv',index = False)
    
    


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
    pos_note_by_hadm_after48.to_csv(notes_dates_path, 'notes_dates_'+ kind+ '.csv',index = False)


def add_amia_cohort(note_stage_path, model_input_path):
    
    note_stage_path = "/data/mimic/PUI_data/data/"

    data = pd.read_csv(note_stage_path + "jamia_note_stage_pu_marked.csv")

    egem_data = pd.read_csv(model_input_path + "/limnote_only_note_threepapers_egem.csv")

    data_merged = data.merge(egem_data, on = 'HADM_ID')


    data_merged['PU_mention_in_both'] = data_merged['note_PU_positive'] & data_merged['Pressure_ulcer'] > 0

    (data_merged['PU_mention_in_both'] == True).sum()

    data_lim = data_merged[[ 'HADM_ID', 'TEXT','ground_truth','PU_mention_in_both']]

    data_lim = data_lim.sort_values(by = 'HADM_ID')

    data_lim['PU_mention_in_both'] = data_lim['PU_mention_in_both'].astype(int)

    data_lim.to_csv(model_input_path + "/limnote_only_note_threepapers_aimia.csv", index = False)


def define_cohorts(data_path, model_input_path, first_date_path, note_stage_path, notes_date_path Pu_marked_charts)

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

        data_path = '/data/mimic/CSVs'
        df_note = pd.read_csv(data_path +'/NOTEEVENTS.csv')
        df_admission = pd.read_csv(data_path +'/ADMISSIONS.csv')
        df_patient = pd.read_csv(data_path +'/PATIENTS.csv')
        staging_df = pd.read_csv("/data/mimic/PU_chart_vivian_ids.csv")
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

        extract_earliest_date(data_path, first_date_path, Pu_marked_charts, notes_date_path kind )
        
        
        cohort_data = first_date_path + 'the_very_first_date_after24_stage_' + kind + '.csv'
        stage = pd.read_csv(cohort_data)
        note = pd.read_csv(notes_date_path, 'notes_dates_'+ kind+ '.csv')
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

    add_amia_cohort(note_stage_path, model_input_path)
    

data_path = '/data/mimic/CSVs'

model_input_path = '/data/mimic/PUI_data/data_test/'

first_date_path = '/data/mimic/PUI_data/data/'

notes_date_path = '/data/mimic/PUI_data/data/'

note_stage_path = "/data/mimic/PUI_data/data/"


Pu_marked_charts = "/data/mimic/PU_chart_vivian_ids.csv"


define_cohorts(data_path, model_input_path, first_date_path, note_stage_path,notes_date_path, Pu_marked_charts)