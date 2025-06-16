import pandas as pd
from collections import Counter

import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

import numpy as np

from multiprocessing import Pool

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

import math

from tqdm import tqdm
import time

import os
import re

import requests
import json

from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed

class ICUDataInput:
    def __init__(self, d_items_path, input_events_path, pro_events_path, output_events_path, 
                 icu_patient_path, d_labitems_path, admission_path, all_patients_path):
        self.d_items_data = None
        self.input_events_data = None
        self.pro_events_data = None
        self.output_event_data = None
        self.ICU_patient_data = None
        self.d_labitems_data = None
        
        # Store file paths
        self.d_items_path = d_items_path
        self.input_events_path = input_events_path
        self.pro_events_path = pro_events_path
        self.output_events_path = output_events_path
        self.icu_patient_path = icu_patient_path
        self.d_labitems_path = d_labitems_path
        self.admission_path = admission_path
        self.all_patients_path = all_patients_path

    def load_data(self):
        self.d_items_data = pd.read_csv(self.d_items_path, compression = 'gzip')
        self.input_events_data = pd.read_csv(self.input_events_path, compression = 'gzip')
        self.pro_events_data = pd.read_csv(self.pro_events_path, compression = 'gzip')
        self.output_event_data = pd.read_csv(self.output_events_path, compression = 'gzip')
        self.ICU_patient_data = pd.read_csv(self.icu_patient_path, compression = 'gzip')
        self.d_labitems_data = pd.read_csv(self.d_labitems_path, compression = 'gzip')
        self.admission_data = pd.read_csv(self.admission_path, compression = 'gzip')
        self.patients_data = pd.read_csv(self.all_patients_path, compression = 'gzip')

    def quick_process_data(self):
        self.icu_ad_list = pd.unique(self.ICU_patient_data["stay_id"])
        self.patient_list = pd.unique(self.ICU_patient_data["subject_id"])

    def get_icu_patient_data(self):
        return self.ICU_patient_data

    def get_icu_ad_list(self):
        return self.icu_ad_list

    def get_patient_list(self):
        return self.patient_list


class VariableSearch:
    def __init__(self, d_items_table, search_column = 'label'):
        self.d_items_data = d_items_table
        self.search_column = search_column
        
    def translate_keyword_openai(self, keyword, target_language, api_key):
        """
        Translate keyword using OpenAI API
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        prompt = f"Translate the following medical term to {target_language}. Only return the translation, no explanation: {keyword}"
        
        data = {
            "model": "gpt-3.5-turbo",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 50,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        except Exception as e:
            print(f"Translation failed: {e}")
            return keyword
    
    def translate_keyword_claude(self, keyword, target_language, api_key):
        """
        Translate keyword using Claude API (Anthropic)
        """
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        prompt = f"Translate the following medical term to {target_language}. Only return the translation, no explanation: {keyword}"
        
        data = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = requests.post(url, headers = headers, json = data)
            response.raise_for_status()
            result = response.json()
            return result['content'][0]['text'].strip()
        except Exception as e:
            print(f"Translation failed: {e}")
            return keyword
    
    def translate_keyword_custom(self, keyword, target_language, translation_function):
        """
        Translate keyword using a custom translation function
        """
        try:
            return translation_function(keyword, target_language)
        except Exception as e:
            print(f"Translation failed: {e}")
            return keyword

    def search_by_keyword(self, keyword, 
                          output_column = 'itemid', use_regex = False, 
                          enable_translation = False, target_languages = None, 
                          translation_service = 'openai', api_key = None, custom_translator = None):
        """
        Enhanced search with multilingual support
        
        Parameters:
        - keyword: Original search keyword
        - output_column: Column to return from filtered results
        - use_regex: Whether to use regular expressions
        - enable_translation: Whether to enable translation
        - target_languages: List of target languages for translation (e.g., ['German', 'French'])
        - translation_service: 'openai', 'claude', or 'custom'
        - api_key: API key for translation service
        - custom_translator: Custom translation function
        """
        
        search_terms = [keyword] 
        
        # Add translated keywords if translation is enabled
        if enable_translation and target_languages:
            for lang in target_languages:
                if translation_service == 'openai' and api_key:
                    translated = self.translate_keyword_openai(keyword, lang, api_key)
                elif translation_service == 'claude' and api_key:
                    translated = self.translate_keyword_claude(keyword, lang, api_key)
                elif translation_service == 'custom' and custom_translator:
                    translated = self.translate_keyword_custom(keyword, lang, custom_translator)
                else:
                    print(f"Translation service not configured properly for {lang}")
                    continue
                
                if translated and translated != keyword:
                    search_terms.append(translated)
                    print(f"Translated '{keyword}' to {lang}: '{translated}'")
        
        # Perform search with all terms (original + translated)
        all_filtered_data = pd.DataFrame()
        
        for term in search_terms:
            if use_regex:
                # Use regular expressions for searching
                pattern = re.compile(term, re.IGNORECASE)
                filtered_data = self.d_items_data[self.d_items_data[self.search_column].str.contains(pattern, na = False)]
            else:
                # Use simple string matching
                filtered_data = self.d_items_data[self.d_items_data[self.search_column].str.contains(term, case = False, na = False)]
            
            all_filtered_data = pd.concat([all_filtered_data, filtered_data], ignore_index=True)
        
        # Remove duplicates and return results
        all_filtered_data = all_filtered_data.drop_duplicates()

        return all_filtered_data[output_column].tolist()
    
    def batch_search_multilingual(self, keywords, target_languages = None, **kwargs):
        """
        Search multiple keywords with translation support
        
        Parameters:
        - keywords: List of keywords to search
        - target_languages: List of target languages
        - **kwargs: Additional parameters for search_by_keyword
        """
        all_results = []
        
        for keyword in keywords:
            results = self.search_by_keyword(
                keyword, 
                target_languages=target_languages,
                **kwargs
            )
            all_results.extend(results)
        
        # Remove duplicates and return
        return list(set(all_results))

class VariableSelect:
    def __init__(self, d_items_table, VitalSigns_id, GCS_score_id, Vent_para_id, Labs_id, General_id, ADT_id, Additional_id_1):
        self.d_items_data = d_items_table

        # Store 'item_id' lists
        self.VitalSigns_id = VitalSigns_id
        self.GCS_score_id = GCS_score_id
        self.Vent_para_id = Vent_para_id
        self.Labs_id = Labs_id
        self.General_id = General_id
        self.ADT_id = ADT_id
        self.add_id_1 = Additional_id_1

    def select_variables(self):
        variable_list = self.VitalSigns_id + self.GCS_score_id + self.Vent_para_id + self.Labs_id + self.General_id + self.ADT_id + self.add_id_1

        self.d_items_data = self.d_items_data[self.d_items_data['itemid'].isin(variable_list)]

    def get_selected_data(self):
        return self.d_items_data
    
    def get_selected_ids(self):
        return self.variable_list
    
    def select_data_chartEvents(self):
        self.d_items_data_chart = self.d_items_data[self.d_items_data['linksto'] == 'chartevents'].copy()
        return self.d_items_data_chart
    
    def select_data_outputEvents(self):
        self.d_items_data_output = self.d_items_data[self.d_items_data['linksto'] == 'outputevents'].copy()
        return self.d_items_data_output
    
    def select_data_datetimeEvents(self):
        self.d_items_data_datetime = self.d_items_data[self.d_items_data['linksto'] == 'datetimeevents'].copy()
        return self.d_items_data_datetime
    
    def select_data_ingredientEvents(self):
        self.d_items_data_ingredient = self.d_items_data[self.d_items_data['linksto'] == 'ingredientevents'].copy()
        return self.d_items_data_ingredient

class ChartEventsProcess:
    def __init__(self, chart_events_path, item_id_list):
        self.chart_events_path = chart_events_path
        self.item_id_list = item_id_list
        self.chart_events_data = None

    def load_and_filter_data(self, var_type):
        self.chart_events_data = dd.read_csv(
            self.chart_events_path,
            dtype = var_type,
            compression = 'gzip',
            assume_missing = True, 
            blocksize = None
        )

        self.chart_events_data = self.chart_events_data[self.chart_events_data.itemid.isin(self.item_id_list)]
        try:
            with ProgressBar():
                self.chart_events_data = self.chart_events_data.compute()
            print("Data successfully loaded!")
        except Exception as e:
            print(f"Error computing DataFrame: {e}")

    def select_columns(self):
        names_select = ['subject_id', 'stay_id', 'itemid', 'charttime', 'value', 'valuenum', 'valueuom']
        self.chart_events_data = self.chart_events_data[names_select]

    def get_chart_events_data(self):
        return self.chart_events_data

    def save_chart_events_data(self, file_path):
        """
        Save the chart_events_data to a specified file path.
        
        Parameters:
        - file_path: The path where the data should be saved.
        """
        if self.chart_events_data is not None:
            self.chart_events_data.to_csv(file_path, index = False)
            print(f"Data successfully saved to {file_path}")
        else:
            print("No data to save. Please load and filter data first!")


class PatientDataProcess:
    def __init__(self, ICU_patient_data):
        self.ICU_patient_data = ICU_patient_data

    def filter_ICU_patients(self, icu_unit_list):
        self.ICU_patient_data = self.ICU_patient_data[self.ICU_patient_data['first_careunit'].isin(icu_unit_list)]
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

    def calculate_los(self):
        self.ICU_patient_data['intime'] = pd.to_datetime(self.ICU_patient_data['intime'])
        self.ICU_patient_data['outtime'] = pd.to_datetime(self.ICU_patient_data['outtime'])
        self.ICU_patient_data['TD_LOS'] = (self.ICU_patient_data['outtime'] - self.ICU_patient_data['intime']).dt.days

    def denote_readmission_cases(self, readmission_observation_days = 30):
        self.ICU_patient_data = self.ICU_patient_data.sort_values(by = ['subject_id', 'intime'])
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

        pa_list = pd.unique(self.ICU_patient_data['subject_id'])
        
        icu_rd_list = []

        for i in range(len(pa_list)):
            sub_data = self.ICU_patient_data[self.ICU_patient_data['subject_id'] == pa_list[i]]
            if len(pd.unique(sub_data['stay_id'])) > 1:
                icu_rd_list.append(pa_list[i])

        ICU_patient_data_rd = self.ICU_patient_data[self.ICU_patient_data['subject_id'].isin(icu_rd_list)].copy()
        pa_list_d = []
        icu_rd_list = []
        disc_fail_list = []

        for i in tqdm(range(len(icu_rd_list))):
            sub_data = ICU_patient_data_rd[ICU_patient_data_rd['subject_id'] == icu_rd_list[i]]
            for j in range(1, len(sub_data)):
                if sub_data['stay_id'].iloc[j] != sub_data['stay_id'].iloc[j-1]:
                    if sub_data['intime'].iloc[j] - sub_data['outtime'].iloc[j-1] <= pd.Timedelta(f'{readmission_observation_days} days 00:00:00'):
                        pa_list_d.append(icu_rd_list[i])
                        disc_fail_list.append(sub_data['stay_id'].iloc[j - 1])
                        icu_rd_list.append(sub_data['stay_id'].iloc[j])
        
                else:
                    print("Error: ", sub_data['stay_id'].iloc[j])

        self.ICU_patient_data[f'discharge_fail_{readmission_observation_days}_day'] = 0
        self.ICU_patient_data[f'readmission_{readmission_observation_days}_day'] = 0

        for stay_id in disc_fail_list:
            self.ICU_patient_data.loc[self.ICU_patient_data['stay_id'] == stay_id, f'discharge_fail_{readmission_observation_days}_day'] = 1

        for stay_id in icu_rd_list:
            self.ICU_patient_data.loc[self.ICU_patient_data['stay_id'] == stay_id, f'readmission_{readmission_observation_days}_day'] = 1
        
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

    def denote_death_cases(self, admission_data, patients_data, readmission_observation_days = 30):

        patients_data_select = patients_data.drop(columns = ['anchor_year', 'anchor_year_group'])
        admission_data_select = admission_data[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type', 'race']]

        patients_data_select = patients_data_select[patients_data_select['subject_id'].isin(self.ICU_patient_data['subject_id'])]
        admission_data_select = admission_data_select[admission_data_select['subject_id'].isin(self.ICU_patient_data['subject_id'])]

        admission_data_select_v1 = admission_data_select[['subject_id', 'race']].copy()
        admission_data_select_v1 = admission_data_select_v1.drop_duplicates(subset = ['subject_id'], keep = 'first')

        self.ICU_patient_data = pd.merge(self.ICU_patient_data, admission_data_select_v1, how = 'left', on = 'subject_id')
        self.ICU_patient_data = pd.merge(self.ICU_patient_data, patients_data_select, how = 'left', on = 'subject_id')
        
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

        self.ICU_patient_data['dod'] = pd.to_datetime(self.ICU_patient_data['dod'])
        self.ICU_patient_data['TD_death_disch'] = self.ICU_patient_data['dod'] - self.ICU_patient_data['outtime']

        self.ICU_patient_data['death_in_ICU'] = 0
        self.ICU_patient_data[f'death_out_ICU_{readmission_observation_days}_day'] = 0

        self.ICU_patient_data.loc[self.ICU_patient_data['TD_death_disch'] <= pd.Timedelta(0), 'death_in_ICU'] = 1
        self.ICU_patient_data.loc[(self.ICU_patient_data['TD_death_disch'] > pd.Timedelta(0)) & 
                                  (self.ICU_patient_data['TD_death_disch'] <= pd.Timedelta(days = readmission_observation_days)), f'death_out_ICU_{readmission_observation_days}_day'] = 1

        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

    def denote_readmission_count(self, readmission_observation_days = 30):
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)
        patient_list = pd.unique(self.ICU_patient_data['subject_id'])

        self.ICU_patient_data[f'readmission_count_{readmission_observation_days}_day'] = 0

        for patient_id in patient_list:
            sub_data = self.ICU_patient_data.loc[self.ICU_patient_data['subject_id'] == patient_id]

            counts = 0

            for idx, row in sub_data.iterrows():

                if row[f'readmission_{readmission_observation_days}_day'] == 1:
                    counts = counts + 1
                    
                else:
                    counts = 0

                self.ICU_patient_data.at[idx, f'readmission_count_{readmission_observation_days}_day'] = counts                      

    def get_ICU_patient_data(self):
        return self.ICU_patient_data
    

class GenerateDataSet:
    def __init__(self, chart_events_data, d_items_data_chart, icu_patient_data):
        self.chart_events_data = chart_events_data
        self.d_items_data_chart = d_items_data_chart
        self.icu_patient_data = icu_patient_data

    def prepare_chart_events_data(self, items_delete_list):
        self.chart_events_data = self.chart_events_data[self.chart_events_data['stay_id'].isin(self.icu_patient_data['stay_id'])]
        self.chart_events_data = self.chart_events_data.reset_index(drop = True)
        self.chart_events_data[['subject_id', 'hadm_id', 'stay_id', 'itemid']] = self.chart_events_data[['subject_id', 'hadm_id', 'stay_id', 'itemid']].astype('int64')
        
        self.d_items_data_chart = self.d_items_data_chart[~self.d_items_data_chart['label'].isin(items_delete_list)]

        self.chart_events_data = self.chart_events_data[self.chart_events_data['itemid'].isin(self.d_items_data_chart['itemid'])]
        self.chart_events_data = self.chart_events_data[self.chart_events_data['stay_id'].isin(self.icu_patient_data['stay_id'])]
        self.chart_events_data = self.chart_events_data.reset_index(drop = True)

        drop_patient_list = pd.unique(self.icu_patient_data[~self.icu_patient_data['stay_id'].isin(self.chart_events_data['stay_id'])]['subject_id'])
        self.icu_patient_data = self.icu_patient_data[~self.icu_patient_data['subject_id'].isin(drop_patient_list)]
        self.icu_patient_data = self.icu_patient_data.reset_index(drop = True)

        drop_patient_list = pd.unique(self.icu_patient_data[self.icu_patient_data['los'].isnull()]['subject_id'])
        self.icu_patient_data = self.icu_patient_data[~self.icu_patient_data['subject_id'].isin(drop_patient_list)]
        self.icu_patient_data = self.icu_patient_data.reset_index(drop = True)

        self.chart_events_data = self.chart_events_data[self.chart_events_data['stay_id'].isin(self.icu_patient_data['stay_id'])]
        self.chart_events_data = self.chart_events_data.reset_index(drop = True)

        self.chart_events_data['charttime'] = pd.to_datetime(self.chart_events_data['charttime'])
        self.chart_events_data['storetime'] = pd.to_datetime(self.chart_events_data['storetime'])
    
    def data_selection(self, data, i_1, i_2, i_3):
        sub_data = data.loc[(data['charttime'] >= i_1) & (data['charttime'] <= i_2) & (data["itemid"] == i_3)]
        return sub_data

    def dataset_generation(self, physio_table):
        icu_stay_list = pd.unique(self.icu_patient_data['stay_id'])
        for i in range(len(icu_stay_list)):
            
            print("The number of processed ICU stay admissions: ", i)
            
            index = self.icu_patient_data["intime"].iloc[i]
            
            s_table_id = self.chart_events_data[self.chart_events_data['stay_id'] == icu_stay_list[i]]

            while index <= self.icu_patient_data["outtime"].iloc[i]:
                physio_table['subject_id'].append(self.icu_patient_data['subject_id'].iloc[i])
                physio_table['hadm_id'].append(self.icu_patient_data['hadm_id'].iloc[i])
                physio_table['stay_id'].append(self.icu_patient_data['stay_id'].iloc[i])
                physio_table['icu_starttime'].append(self.icu_patient_data['intime'].iloc[i])
                physio_table['icu_endtime'].append(self.icu_patient_data['outtime'].iloc[i]) 
                physio_table['los'].append(self.icu_patient_data['los'].iloc[i])        
                physio_table['discharge_fail'].append(self.icu_patient_data['discharge_fail_30_day'].iloc[i])
                physio_table['readmission'].append(self.icu_patient_data['readmission_30_day'].iloc[i])
                physio_table['readmission_count'].append(self.icu_patient_data['readmission_count_30_day'].iloc[i])
                physio_table['death_in_ICU'].append(self.icu_patient_data['death_in_ICU'].iloc[i])
                physio_table['death_out_ICU'].append(self.icu_patient_data['death_out_ICU_30_day'].iloc[i])
                physio_table['age'].append(self.icu_patient_data['anchor_age'].iloc[i])
                physio_table['gender'].append(self.icu_patient_data['gender'].iloc[i])
                physio_table['race'].append(self.icu_patient_data['race'].iloc[i])
                
                td = pd.Timedelta('0 days 12:00:00')
                rd_idx = physio_table['readmission_count'][-1]
                
                if rd_idx <= 4:
                
                    index_1 = index + td * (0.5**rd_idx)

                    if index_1 <= self.icu_patient_data["outtime"].iloc[i]:
                        physio_table['time'].append(index_1)
                    else:
                        index_1 = self.icu_patient_data["outtime"].iloc[i]
                        physio_table['time'].append(index_1)

                    for j in range(len(self.d_items_data_chart)):
                        s_table = self.data_selection(s_table_id, index, index_1, self.d_items_data_chart["itemid"].iloc[j])

                        n = len(s_table)

                        if n >= 1:
                            physio_table[self.d_items_data_chart['label'].iloc[j]].append(s_table['valuenum'].mean())
                            # physio_table[d_items_data_chart_select['label'].iloc[j]].append(s_table['valuenum'].iloc[-1])

                        else:
                            physio_table[self.d_items_data_chart['label'].iloc[j]].append(np.nan)

                    index = index + td * (0.5**rd_idx)
                    
                else:
                    rd_idx = 4
                    index_1 = index + td * (0.5**rd_idx)

                    if index_1 <= self.icu_patient_data["outtime"].iloc[i]:
                        physio_table['time'].append(index_1)
                    else:
                        index_1 = self.icu_patient_data["outtime"].iloc[i]
                        physio_table['time'].append(index_1)

                    for j in range(len(self.d_items_data_chart)):
                        s_table = self.data_selection(s_table_id, index, index_1, self.d_items_data_chart["itemid"].iloc[j])

                        n = len(s_table)

                        if n >= 1:
                            physio_table[self.d_items_data_chart['label'].iloc[j]].append(s_table['valuenum'].mean())
                            # physio_table[d_items_data_chart_select['label'].iloc[j]].append(s_table['valuenum'].iloc[-1])

                        else:
                            physio_table[self.d_items_data_chart['label'].iloc[j]].append(np.nan)

                    index = index + td * (0.5**rd_idx)
        
        self.generated_dataset = pd.DataFrame.from_dict(physio_table)

    def assign_blood_pressure(self, row):
        if pd.isna(row['Arterial Blood Pressure systolic']) and not pd.isna(row['Non Invasive Blood Pressure systolic']):
            return row['Non Invasive Blood Pressure systolic']
        elif not pd.isna(row['Arterial Blood Pressure systolic']):
            return row['Arterial Blood Pressure systolic']
        elif not pd.isna(row['ART BP Systolic']):
            return row['ART BP Systolic']
        else:
            return np.nan
    
    def assign_blood_pressure_diastolic(self, row):
        if pd.isna(row['Arterial Blood Pressure diastolic']) and not pd.isna(row['Non Invasive Blood Pressure diastolic']):
            return row['Non Invasive Blood Pressure diastolic']
        elif not pd.isna(row['Arterial Blood Pressure diastolic']):
            return row['Arterial Blood Pressure diastolic']
        elif not pd.isna(row['ART BP Diastolic']):
            return row['ART BP Diastolic']
        else:
            return np.nan

    def assign_blood_pressure_mean(self, row):
        if pd.isna(row['Arterial Blood Pressure mean']) and not pd.isna(row['Non Invasive Blood Pressure mean']):
            return row['Non Invasive Blood Pressure mean']
        elif not pd.isna(row['Arterial Blood Pressure mean']):
            return row['Arterial Blood Pressure mean']
        elif not pd.isna(row['ART BP Mean']):
            return row['ART BP Mean']
        else:
            return np.nan

    def assign_temperature(self, row):
        if pd.isna(row['Temperature Celsius']) and not pd.isna(row['Temperature Fahrenheit']):
            return (row['Temperature Fahrenheit']-32) * 5.0/9.0
        elif not pd.isna(row['Temperature Celsius']):
            return row['Temperature Celsius']
        else:
            return np.nan

    def assign_SaO2(self, row):
        if pd.isna(row['Arterial O2 Saturation']) and not pd.isna(row['O2 saturation pulseoxymetry']):
            return row['O2 saturation pulseoxymetry']
        elif not pd.isna(row['Arterial O2 Saturation']):
            return row['Arterial O2 Saturation']
        else:
            return np.nan

    def assign_gcs_score(self, row):
        return row['GCS - Eye Opening'] + row['GCS - Verbal Response'] + row['GCS - Motor Response']

    def assign_peep_level(self, row):
        if pd.isna(row['PEEP set']) and not pd.isna(row['Total PEEP Level']):
            return row['Total PEEP Level']
        elif not pd.isna(row['PEEP set']):
            return row['PEEP set']
        else:
            return np.nan

    def assign_weight(self, row):
        if not pd.isna(row['Daily Weight']):
            return row['Daily Weight']
        elif not pd.isna(row['Admission Weight (Kg)']):
            return row['Admission Weight (Kg)']
        elif not pd.isna(row['Admission Weight (lbs.)']):
            return row['Admission Weight (lbs.)'] * 0.453592  # Convert lbs to kg
        else:
            return np.nan
    
    def assign_weight_2(self, row):
        if not pd.isna(row['Weight']):
            return row['Weight']
        elif not pd.isna(row['patientweight']):
            return row['patientweight']
        else:
            return np.nan

    def process_gen_data(self, pro_events_data, drop_columns = []):
        self.generated_dataset['Tidal Volume (set)'] = self.generated_dataset['Tidal Volume (set)']/1000
        self.generated_dataset['Tidal Volume (observed)'] = self.generated_dataset['Tidal Volume (observed)']/1000
        self.generated_dataset['Tidal Volume (spontaneous)'] = self.generated_dataset['Tidal Volume (spontaneous)']/1000

        gender_dummies = pd.get_dummies(self.generated_dataset['gender'])
        self.generated_dataset = pd.concat([self.generated_dataset, gender_dummies], axis = 'columns')
        self.generated_dataset = self.generated_dataset.drop(['gender', 'F'], axis = 'columns')

        self.generated_dataset = self.generated_dataset.drop(columns = ['race'])

        icu_stayid_list = self.generated_dataset['stay_id'].unique()

        self.generated_dataset['discharge_action'] = 0

        for i in range(len(icu_stayid_list)):
            time_idx = self.generated_dataset[(self.generated_dataset['stay_id'] == icu_stayid_list[i])]['time'].iloc[-1]
            self.generated_dataset.loc[(self.generated_dataset['stay_id'] == icu_stayid_list[i]) & (self.generated_dataset['time'] == time_idx), 'discharge_action'] = 1

        self.generated_dataset['Blood Pressure Systolic'] = self.generated_dataset.apply(self.assign_blood_pressure, axis = 1)
        self.generated_dataset['Blood Pressure Diastolic'] = self.generated_dataset.apply(self.assign_blood_pressure_diastolic, axis = 1)
        self.generated_dataset['Blood Pressure Mean'] = self.generated_dataset.apply(self.assign_blood_pressure_mean, axis = 1)

        self.generated_dataset['Temperature Celsius'] = self.generated_dataset.apply(self.assign_temperature, axis = 1)
        self.generated_dataset['SaO2'] = self.generated_dataset.apply(self.assign_SaO2, axis = 1)

        self.generated_dataset['GCS Score'] = self.generated_dataset.apply(self.assign_gcs_score, axis = 1)
        self.generated_dataset['PEEP Level'] = self.generated_dataset.apply(self.assign_peep_level, axis = 1)
        self.generated_dataset['Weight'] = self.generated_dataset.apply(self.assign_weight, axis = 1)

        self.generated_dataset = self.generated_dataset.drop(columns = drop_columns)
        pro_events_data_weight = pro_events_data[['stay_id', 'patientweight']]
        pro_events_data_weight = pro_events_data_weight.drop_duplicates(subset = ['stay_id'], keep = 'first')

        self.generated_dataset = pd.merge(self.generated_dataset, pro_events_data_weight, on = 'stay_id', how = 'left')
        self.generated_dataset['weight'] = self.generated_dataset.apply(self.assign_weight_2, axis = 1)
        self.generated_dataset = self.generated_dataset.drop(columns = ['Weight', 'patientweight'])

    def abnormal_data_filter(self, method = 'iqr', iqr_factor = 1.5, z_threshold = 3.0, abnormal_var_list = []):
        """
        Filter abnormal data using IQR or Z-score method
        
        Parameters:
        - method: 'iqr' or 'zscore' (default: 'iqr')
        - iqr_factor: Factor for IQR method (default: 1.5)
        - z_threshold: Threshold for Z-score method (default: 3.0)
        - abnormal_var_list: List of variables to apply filtering on
        """
        if not abnormal_var_list:
            print("Warning: No variables specified for abnormal data filtering")
            return
            
        # Check if all variables exist in the dataset
        missing_vars = [var for var in abnormal_var_list if var not in self.generated_dataset.columns]
        if missing_vars:
            print(f"Warning: The following variables are not found in the dataset: {missing_vars}")
            abnormal_var_list = [var for var in abnormal_var_list if var in self.generated_dataset.columns]
        
        if not abnormal_var_list:
            print("No valid variables found for filtering")
            return
            
        abv_data = self.generated_dataset[abnormal_var_list]
        physio_df_v2_abn = self.generated_dataset.copy()

        if method.lower() == 'iqr':
            # IQR method
            print(f"Applying IQR method with factor {iqr_factor}")
            
            q1 = abv_data.quantile(0.25)
            q3 = abv_data.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - iqr_factor * iqr
            upper_bound = q3 + iqr_factor * iqr
            
            for column in abv_data.columns:
                outlier_mask = (physio_df_v2_abn[column] > upper_bound[column]) | (physio_df_v2_abn[column] < lower_bound[column])
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    print(f"IQR: Removing {outlier_count} outliers from {column}")
                physio_df_v2_abn.loc[outlier_mask, column] = np.nan
                
        elif method.lower() == 'zscore':
            # Z-score method
            print(f"Applying Z-score method with threshold {z_threshold}")
            
            for column in abv_data.columns:
                if abv_data[column].notna().sum() > 0:  # Check if column has non-null values
                    mean_val = abv_data[column].mean()
                    std_val = abv_data[column].std()
                    
                    if std_val > 0:  # Avoid division by zero
                        z_scores = np.abs((physio_df_v2_abn[column] - mean_val) / std_val)
                        outlier_mask = z_scores > z_threshold
                        outlier_count = outlier_mask.sum()
                        if outlier_count > 0:
                            print(f"Z-score: Removing {outlier_count} outliers from {column}")
                        physio_df_v2_abn.loc[outlier_mask, column] = np.nan
                    else:
                        print(f"Warning: {column} has zero standard deviation, skipping Z-score filtering")
                else:
                    print(f"Warning: {column} has no valid values, skipping filtering")
        else:
            raise ValueError("Method must be either 'iqr' or 'zscore'")

        # Apply domain-specific filtering (existing logic)
        if 'Inspired O2 Fraction' in physio_df_v2_abn.columns:
            id_delete_list = list(physio_df_v2_abn[physio_df_v2_abn['Inspired O2 Fraction'] > 100]['subject_id'])
            if id_delete_list:
                print(f"Removing {len(id_delete_list)} subjects with Inspired O2 Fraction > 100%")
                physio_df_v2_abn = physio_df_v2_abn[~physio_df_v2_abn['subject_id'].isin(id_delete_list)]
        
        self.generated_dataset = physio_df_v2_abn.copy()
        print(f"Abnormal data filtering completed using {method.upper()} method")

    def save_to_csv(self, file_path):
        self.generated_dataset.to_csv(file_path, index = False)


class PatientDataImputation:
    def __init__(self, generated_dataset):
        self.generated_dataset = generated_dataset.copy()

    def drop_columns(self, var_list = []):
        self.generated_dataset = self.generated_dataset.drop(columns = var_list)

    def drop_missing_columns(self, var_list = [], missing_threshold = 0.75):
        drop_list = []
        for i in var_list:
            if (self.generated_dataset[i].isnull().sum()/len(self.generated_dataset)) > missing_threshold:
                drop_list.append(i)
        
        return drop_list

    def forward_fill_missing_values(self, var_list = []):
        for i in range(len(var_list)):
            self.generated_dataset[var_list[i]] = self.generated_dataset.groupby(by = ['stay_id', 'readmission_count'])[var_list[i]].ffill()

    def linear_impute_missing_values(self, var_list = []):
        for i in range(len(var_list)):
            self.generated_dataset[var_list[i]] = self.generated_dataset.groupby(by = ['stay_id', 'readmission_count'])[var_list[i]].apply(lambda x: x.interpolate(method = 'linear'))

    def process_chunk(self, chunk, imputer):
        chunk_imputed = imputer.fit_transform(chunk)  
        return chunk_imputed

    def knn_impute_missing_values(self, num_neigh = 5, scaler = MinMaxScaler(), chunk_size = 10000, num_jobs = 60):

        num_threads = os.cpu_count()
        print(f"Available CPU threads: {num_threads}")

        if num_jobs > num_threads:
            print(f"Warning: Number of jobs ({num_jobs}) exceeds available CPU threads ({num_threads})")
            user_input = input(f"Please enter a number between 1 and {num_threads}: ")
            chosen = int(user_input)
            if 1 <= chosen <= num_threads:
                num_jobs = chosen
            else:
                print("Invalid input. Using default number of jobs.")
                num_jobs = num_threads - 1
        
        print(f"Using {num_jobs} CPU threads for KNN imputation")

        imputer = KNNImputer(n_neighbors = num_neigh)

        self.generated_dataset = self.generated_dataset.reset_index(drop = True)

        columns_with_missing_values = self.generated_dataset.columns[self.generated_dataset.isnull().any()].tolist()
        self.generated_dataset_pre = self.generated_dataset[columns_with_missing_values].copy()

        self.generated_dataset_pre[columns_with_missing_values] = scaler.fit_transform(self.generated_dataset_pre[columns_with_missing_values])

        chunks = [self.generated_dataset_pre.iloc[i:i + chunk_size] for i in range(0, len(self.generated_dataset_pre), chunk_size)]

        results = Parallel(n_jobs = num_jobs)(
            delayed(self.process_chunk)(chunk, imputer) 
            for chunk in tqdm(chunks, desc = "KNN Imputation Progress")
        )

        self.generated_dataset_pre[columns_with_missing_values] = pd.concat(
            [pd.DataFrame(result, columns = columns_with_missing_values) for result in results],
            ignore_index = True
        )

        self.generated_dataset_pre[columns_with_missing_values] = scaler.inverse_transform(self.generated_dataset_pre[columns_with_missing_values])
        self.generated_dataset = self.generated_dataset.reset_index(drop = True)
        self.generated_dataset[columns_with_missing_values] = self.generated_dataset_pre[columns_with_missing_values]
    
    def save_to_csv(self, file_path):
        self.generated_dataset.to_csv(file_path, index = False)

class StateSpaceBuilder:
    def __init__(self, generated_dataset):
        self.generated_dataset = generated_dataset.copy()
        self.rl_cont_state_table = None
        self.state_id_table = None

    def drop_duplicate_rows(self):
        self.generated_dataset = self.generated_dataset.drop_duplicates()
        m = self.generated_dataset[self.generated_dataset['discharge_action'] == 1]
        duplicates = m[m.duplicated(subset=['stay_id'])]
        if len(duplicates) > 0:
            print(f"Attention!!! Found {len(duplicates)} duplicate rows in the dataset")
            self.generated_dataset = self.generated_dataset.drop(duplicates.index)
        else:
            print("No duplicate rows found in the dataset")
        
        self.generated_dataset = self.generated_dataset.reset_index(drop = True)

    def columns_manipulation(self):
        # initialize RR and TV as NaN
        self.generated_dataset['RR'] = np.nan
        self.generated_dataset['TV'] = np.nan

        # condition 1: Tidal Volume (spontaneous) not equal to 0
        mask_spont_tv = self.generated_dataset['Tidal Volume (spontaneous)'] != 0
        self.generated_dataset.loc[mask_spont_tv, 'TV'] = self.generated_dataset.loc[mask_spont_tv, 'Tidal Volume (spontaneous)']

        # condition 2: Respiratory Rate (spontaneous) not equal to 0
        mask_spont_rr = self.generated_dataset['Respiratory Rate (spontaneous)'] != 0
        self.generated_dataset.loc[mask_spont_tv & mask_spont_rr, 'RR'] = self.generated_dataset.loc[mask_spont_tv & mask_spont_rr, 'Respiratory Rate (spontaneous)']
        self.generated_dataset.loc[mask_spont_tv & ~mask_spont_rr, 'RR'] = self.generated_dataset.loc[mask_spont_tv & ~mask_spont_rr, 'Respiratory Rate']

        # condition 3: Tidal Volume (spontaneous) equal to 0
        mask_observed_tv = self.generated_dataset['Tidal Volume (spontaneous)'] == 0
        self.generated_dataset.loc[mask_observed_tv & mask_spont_rr, 'TV'] = self.generated_dataset.loc[mask_observed_tv & mask_spont_rr, 'Tidal Volume (observed)']
        self.generated_dataset.loc[mask_observed_tv & mask_spont_rr, 'RR'] = self.generated_dataset.loc[mask_observed_tv & mask_spont_rr, 'Respiratory Rate (spontaneous)']

        # condition 4: Tidal Volume (spontaneous) and Respiratory Rate (spontaneous) both equal to 0
        self.generated_dataset.loc[mask_observed_tv & ~mask_spont_rr, 'TV'] = self.generated_dataset.loc[mask_observed_tv & ~mask_spont_rr, 'Tidal Volume (spontaneous)']
        self.generated_dataset.loc[mask_observed_tv & ~mask_spont_rr, 'RR'] = self.generated_dataset.loc[mask_observed_tv & ~mask_spont_rr, 'Respiratory Rate (spontaneous)']

        self.generated_dataset = self.generated_dataset.drop(columns = ['Respiratory Rate', 'Respiratory Rate (spontaneous)', 'Respiratory Rate (Set)', 'Respiratory Rate (Total)', 
                                                                        'Tidal Volume (spontaneous)', 'Tidal Volume (set)', 'Tidal Volume (observed)']).copy()
        
    def icu_discharge_data_selection(self, los_threshold = 15.0):
        self.generated_dataset = self.generated_dataset[~self.generated_dataset['subject_id'].isin(pd.unique(self.generated_dataset[self.generated_dataset['los'] > los_threshold]['subject_id']))].copy()
        self.generated_dataset = self.generated_dataset.reset_index(drop = True)

        drop_patient_list = self.generated_dataset.loc[self.generated_dataset['readmission_count'] >= 6, 'subject_id'].tolist()
        self.generated_dataset = self.generated_dataset[~self.generated_dataset['subject_id'].isin(drop_patient_list)].copy()
        self.generated_dataset = self.generated_dataset.reset_index(drop = True)

        self.generated_dataset['epoch'] = self.generated_dataset.groupby(['stay_id', 'readmission_count']).cumcount() + 1

        self.generated_dataset['id_delete'] = 0.0
        condition_1 = (self.generated_dataset['readmission_count'] == 0) & (self.generated_dataset['death_in_ICU'] == 1)
        self.generated_dataset.loc[condition_1, 'id_delete'] = 1.0
        self.generated_dataset = self.generated_dataset[self.generated_dataset['id_delete'] != 1].copy()
        self.generated_dataset = self.generated_dataset.reset_index(drop = True)

        self.generated_dataset['qsofa'] = self.generated_dataset.apply(self.compute_qsofa, axis = 1)

    def table_split(self, var_outcome = [], var_physio = []):
        self.rl_cont_state_table = self.generated_dataset[var_physio].copy()
        self.state_id_table = self.generated_dataset[var_outcome].copy()

    def discharge_cost_set(self, scaler = MinMaxScaler()):
        # mortality risk costs
        self.state_id_table['death'] = 0.0
        condition = (self.state_id_table['death_in_ICU'] == 1) | (self.state_id_table['death_out_ICU'] == 1)
        self.state_id_table.loc[condition, 'death'] = 1

        self.state_id_table['mortality_costs'] = 0
        condition_1 = (self.state_id_table['discharge_action'] == 1) & (self.state_id_table['death'] == 1)
        self.state_id_table.loc[condition_1, 'mortality_costs'] = 1

        condition_2 = (self.state_id_table['discharge_action'] == 1) & (self.state_id_table['death'] != 1)
        self.state_id_table.loc[condition_2, 'mortality_costs'] = 0

        # readmission risk costs
        self.state_id_table['discharge_fail_costs'] = 0

        condition_1 = (self.state_id_table['discharge_action'] == 1) & (self.state_id_table['discharge_fail'] == 1)
        self.state_id_table.loc[condition_1, 'discharge_fail_costs'] = 1

        condition_2 = (self.state_id_table['discharge_action'] == 1) & (self.state_id_table['discharge_fail'] != 1)
        self.state_id_table.loc[condition_2, 'discharge_fail_costs'] = 0

        # length-of-stay costs
        self.state_id_table['time'] = pd.to_datetime(self.state_id_table['time'])
        self.state_id_table['icu_starttime'] = pd.to_datetime(self.state_id_table['icu_starttime'])
        self.state_id_table['icu_endtime'] = pd.to_datetime(self.state_id_table['icu_endtime'])

        self.state_id_table['los_costs'] = 0.0

        discharge_action_zero_mask = self.state_id_table['discharge_action'] == 0
        self.state_id_table.loc[discharge_action_zero_mask, 'los_costs'] = 12.0 * (0.5 ** np.minimum(self.state_id_table.loc[discharge_action_zero_mask, 'readmission_count'], 4))

        self.state_id_table['los_costs_scaled'] = 0
        self.state_id_table[['los_costs_scaled']] = scaler.fit_transform(self.state_id_table[['los_costs']])

    def qSOFA_safe_action_space(self):
        safe_condition = (self.state_id_table['qSOFA'] == 0) | (self.state_id_table['qSOFA'] == 1)
        unsafe_condition = (self.state_id_table['qSOFA'] == 2) | (self.state_id_table['qSOFA'] == 3)

        self.state_id_table.loc[safe_condition, 'qsofa_safe_action'] = 1.0
        self.state_id_table.loc[unsafe_condition, 'qsofa_safe_action'] = 0.0

    def train_val_test_split(self, scaler = MinMaxScaler(), test_prop = 0.2, val_prop = 0.5, random_seed = 42):
        self.rl_cont_state_table.rename(columns = {'RR': 'Respiratory Rate', 'TV': 'Tidal Volume'}, inplace = True)
        self.rl_cont_state_table['age'] = self.rl_cont_state_table['age'].astype(float)
        self.rl_cont_state_table['M'] = self.rl_cont_state_table['M'].astype(float)
        self.rl_cont_state_table['readmission_count'] = self.rl_cont_state_table['readmission_count'].astype(float)

        self.state_id_table['discharge_action'] = self.state_id_table['discharge_action'].astype(float)
        self.state_id_table['discharge_fail'] = self.state_id_table['discharge_fail'].astype(float)
        self.state_id_table['mortality_costs'] = self.state_id_table['mortality_costs'].astype(float)
        self.state_id_table['discharge_fail_costs'] = self.state_id_table['discharge_fail_costs'].astype(float)
        self.state_id_table['los_costs'] = self.state_id_table['los_costs'].astype(float)
        self.state_id_table['los_costs_scaled'] = self.state_id_table['los_costs_scaled'].astype(float)

        var_list = self.rl_cont_state_table.columns.tolist()
        self.rl_cont_state_table_scaled = self.rl_cont_state_table.copy()
        self.rl_cont_state_table_scaled[var_list] = scaler.fit_transform(self.rl_cont_state_table[var_list])
        self.rl_cont_state_table_scaled['M'] = self.rl_cont_state_table['M'].copy()
        self.rl_cont_state_table_scaled['readmission_count_original'] = self.rl_cont_state_table['readmission_count'].copy()

        condition = (self.rl_cont_state_table_scaled['readmission_count'] == 0.6000000000000001)
        self.rl_cont_state_table_scaled.loc[condition, 'readmission_count'] = 0.6

        train_subject_id, temp_subject_id = train_test_split(pd.unique(self.state_id_table['subject_id']), test_size = test_prop,  random_state = random_seed)
        val_subject_id, test_subject_id = train_test_split(temp_subject_id, test_size = val_prop, random_state = random_seed)

        self.id_table_train = self.state_id_table[self.state_id_table['subject_id'].isin(train_subject_id.tolist())].copy()
        mv_train_index = self.id_table_train.index
        id_index_list = mv_train_index.tolist()

        self.rl_table_train = self.rl_cont_state_table.loc[id_index_list].copy()
        self.rl_table_train_scaled = self.rl_cont_state_table_scaled.loc[id_index_list].copy()

        self.id_table_val = self.state_id_table[self.state_id_table['subject_id'].isin(val_subject_id.tolist())].copy()
        mv_val_index = self.id_table_val.index
        id_index_list = mv_val_index.tolist()

        self.rl_table_val = self.rl_cont_state_table.loc[id_index_list].copy()
        self.rl_table_val_scaled = self.rl_cont_state_table_scaled.loc[id_index_list].copy()

        self.id_table_test = self.state_id_table[self.state_id_table['subject_id'].isin(test_subject_id.tolist())].copy()
        mv_test_index = self.id_table_test.index
        id_index_list = mv_test_index.tolist()

        self.rl_table_test = self.rl_cont_state_table.loc[id_index_list].copy()
        self.rl_table_test_scaled = self.rl_cont_state_table_scaled.loc[id_index_list].copy()        

    def compute_qsofa(self, row):
        score = 0
    
        if row['RR'] >= 22:
            score += 1
        
        if row['Blood Pressure Systolic'] <= 100:
            score += 1
        
        if row['GCS score'] < 15:
            score += 1
    
        return score
    
    def save_to_csv(self, dataset, file_path):
        dataset.to_csv(file_path, index = False)
    


        