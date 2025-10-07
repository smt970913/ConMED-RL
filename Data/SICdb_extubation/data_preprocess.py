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
import pickle

from threadpoolctl import threadpool_limits
from joblib import Parallel, delayed

class ICUDataInput:
    def __init__(self, cases_path, d_references_path, laboratory_path, medication_path, 
                 data_float_h_path, data_range_path, data_ref_path, data_compression='gzip'):
        # main data
        self.cases_data = None  # icu_stays in MIMIC-IV
        self.d_references_data = None  
        self.laboratory_data = None  
        self.medication_data = None  
        self.data_float_h_data = None  
        self.data_range_data = None  
        self.data_ref_data = None  
        
        self.cases_path = cases_path
        self.d_references_path = d_references_path
        self.laboratory_path = laboratory_path
        self.medication_path = medication_path
        self.data_float_h_path = data_float_h_path
        self.data_range_path = data_range_path
        self.data_ref_path = data_ref_path
        self.data_compression = data_compression

    def load_data(self):
        if self.cases_path is not None:
            self.cases_data = pd.read_csv(self.cases_path, compression = self.data_compression)
        if self.d_references_path is not None:
            self.d_references_data = pd.read_csv(self.d_references_path, compression = self.data_compression)
        if self.laboratory_path is not None:
            self.laboratory_data = pd.read_csv(self.laboratory_path, compression = self.data_compression)
        if self.medication_path is not None:
            self.medication_data = pd.read_csv(self.medication_path, compression = self.data_compression)
        if self.data_float_h_path is not None:
            self.data_float_h_data = pd.read_csv(self.data_float_h_path, compression = self.data_compression)
        if self.data_range_path is not None:
            self.data_range_data = pd.read_csv(self.data_range_path, compression = self.data_compression)
        if self.data_ref_path is not None:
            self.data_ref_data = pd.read_csv(self.data_ref_path, compression = self.data_compression)

    def quick_process_data(self):
        self.cases_list = pd.unique(self.cases_data["CaseID"])
        self.patient_list = pd.unique(self.cases_data["PatientID"])

    def get_cases_data(self):
        return self.cases_data

    def get_cases_list(self):
        return self.cases_list

    def get_patient_list(self):
        return self.patient_list


class VariableSearch:
    def __init__(self, d_items_table, search_column = 'label', id_column = 'itemid'):
        self.d_items_data = d_items_table
        self.search_column = search_column
        self.id_column = id_column
        self.last_search_results = None
        self.last_keyword = None
        self.last_api_service = None
        self.last_api_key = None
        
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
            response = requests.post(url, headers = headers, json = data)
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
            "model": "claude-sonnet-4-20250514",
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

    def _call_llm_api(self, prompt, api_service, api_key, max_tokens = 2000):
        """
        Generic method to call LLM API (OpenAI or Claude)
        """
        try:
            if api_service.lower() == 'openai':
                url = "https://api.openai.com/v1/chat/completions"
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                data = {
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.3
                }
                response = requests.post(url, headers = headers, json = data)
                response.raise_for_status()
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
                
            elif api_service.lower() == 'claude':
                url = "https://api.anthropic.com/v1/messages"
                headers = {
                    "x-api-key": api_key,
                    "Content-Type": "application/json",
                    "anthropic-version": "2023-06-01"
                }
                data = {
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": max_tokens,
                    "messages": [{"role": "user", "content": prompt}]
                }
                response = requests.post(url, headers = headers, json = data)
                response.raise_for_status()
                result = response.json()
                return result['content'][0]['text'].strip()
            else:
                return "Unsupported API service"
                
        except Exception as e:
            return f"API call failed: {e}"

    def _analyze_search_column_variables(self, keyword, api_service, api_key):
        """
        Use LLM to analyze which variables in search_column might be related to the keyword
        """
        unique_values = self.d_items_data[self.search_column].unique()
        # Limit the number of values to analyze to avoid API limits
        unique_values = unique_values[:100] if len(unique_values) > 100 else unique_values
        
        prompt = f"""
        As a medical data expert, analyze the following list of medical variables and identify which ones might be related to the search keyword: "{keyword}".

        Variables to analyze:
        {', '.join([str(v) for v in unique_values if pd.notna(v)])}

        Please:
        1. List the variables that are most likely related to "{keyword}"
        2. Provide a brief explanation for each selected variable
        3. Rate the relevance on a scale of 1-5 (5 being most relevant)
        4. Format your response as:
           - Variable Name (Relevance: X/5): Brief explanation

        Focus on medical relevance and semantic similarity.
        """
        
        response = self._call_llm_api(prompt, api_service, api_key, max_tokens = 2000)
        return response

    def _analyze_unit_columns(self, keyword, api_service, api_key):
        """
        Analyze unit columns to find relevant units related to the keyword
        """
        # Find columns containing 'unit' in their name
        unit_columns = [col for col in self.d_items_data.columns if 'unit' in col.lower()]
        
        if not unit_columns:
            return "No unit columns found in the dataset."
        
        # Get unique values from unit columns
        all_units = set()
        for col in unit_columns:
            units = self.d_items_data[col].dropna().unique()
            all_units.update(units)
        
        # Limit units to analyze
        all_units = list(all_units)[:50] if len(all_units) > 50 else list(all_units)
        
        prompt = f"""
        As a medical data expert, analyze the following list of measurement units and identify which ones might be related to the search keyword: "{keyword}".

        Available unit columns: {', '.join(unit_columns)}
        
        Units to analyze:
        {', '.join([str(u) for u in all_units if str(u) != 'nan'])}

        Please:
        1. List the units that are most likely related to "{keyword}"
        2. Explain why each unit is relevant to the keyword
        3. Rate the relevance on a scale of 1-5 (5 being most relevant)
        4. Format your response as:
           - Unit (Relevance: X/5): Brief explanation of relevance

        Focus on medical measurement context and relevance to the keyword.
        """
        
        response = self._call_llm_api(prompt, api_service, api_key, max_tokens = 2000)
        return response

    def _get_variables_by_units(self, selected_units):
        """
        Get variables that use the selected units
        """
        unit_columns = [col for col in self.d_items_data.columns if 'unit' in col.lower()]
        if not unit_columns:
            return []
        
        related_items = []
        for unit_col in unit_columns:
            mask = self.d_items_data[unit_col].isin(selected_units)
            related_data = self.d_items_data[mask]
            related_items.extend(related_data[self.id_column].tolist())
        
        return list(set(related_items))

    def _interactive_chatbot_mode(self, keyword, api_service, api_key):
        """
        Interactive chatbot mode for enhanced search
        """
        print("\n" + "="*60)
        print("🤖 ENHANCED SEARCH ASSISTANT")
        print("="*60)
        print(f"I see you're looking for variables related to: '{keyword}'")
        print("I can help you find more relevant variables using AI analysis.")
        print("\nPlease choose an option:")
        print("1. 📊 Analyze all available variable names for relevance")
        print("2. 🔬 Search by measurement units")
        print("3. 💬 Chat with assistant (ask questions about the search)")
        print("4. ❌ Exit enhanced search")
        
        while True:
            try:
                choice = input("\nEnter your choice (1-4): ").strip()
                
                if choice == '1':
                    print("\n🔍 Analyzing variable names...")
                    analysis = self._analyze_search_column_variables(keyword, api_service, api_key)
                    print("\n📊 VARIABLE ANALYSIS RESULTS:")
                    print("-" * 40)
                    print(analysis)
                    
                    # Ask if user wants to add these to search results
                    add_to_search = input("\nWould you like to search for specific variables mentioned above? (y/n): ").lower()
                    if add_to_search == 'y':
                        manual_vars = input("Enter variable names (comma-separated): ").strip()
                        if manual_vars:
                            additional_results = self._search_manual_variables(manual_vars.split(','))
                            print(f"\n✅ Found {len(additional_results)} additional items from manual search")
                            return additional_results
                    
                elif choice == '2':
                    print("\n🔬 Analyzing measurement units...")
                    unit_analysis = self._analyze_unit_columns(keyword, api_service, api_key)
                    print("\n🔬 UNIT ANALYSIS RESULTS:")
                    print("-" * 40)
                    print(unit_analysis)
                    
                    # Ask if user wants to search by units
                    search_by_units = input("\nWould you like to search by specific units? (y/n): ").lower()
                    if search_by_units == 'y':
                        units_input = input("Enter unit names (comma-separated): ").strip()
                        if units_input:
                            selected_units = [u.strip() for u in units_input.split(',')]
                            unit_results = self._get_variables_by_units(selected_units)
                            print(f"\n✅ Found {len(unit_results)} items with selected units")
                            return unit_results
                    
                elif choice == '3':
                    print("\n💬 Chat mode activated! Ask me anything about your search.")
                    print("Type 'exit' to return to the main menu.")
                    
                    while True:
                        user_question = input("\nYour question: ").strip()
                        if user_question.lower() == 'exit':
                            break
                        
                        chat_prompt = f"""
                        You are a medical data search assistant. The user is looking for variables related to '{keyword}' in a medical database.
                        
                        User's question: {user_question}
                        
                        Please provide helpful information about:
                        - Medical terminology related to '{keyword}'
                        - Common measurement units or parameters
                        - Alternative search terms or synonyms
                        - General guidance for medical data search
                        
                        Keep your response concise and focused on helping the user find relevant variables.
                        """
                        
                        response = self._call_llm_api(chat_prompt, api_service, api_key, max_tokens = 2000)
                        print(f"\n🤖 Assistant: {response}")
                
                elif choice == '4':
                    print("\n👋 Exiting enhanced search mode...")
                    return []
                
                else:
                    print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting enhanced search mode...")
                return []
            except Exception as e:
                print(f"❌ Error: {e}")

    def _search_manual_variables(self, variable_names):
        """
        Search for specific variable names manually entered by user
        """
        results = []
        for var_name in variable_names:
            var_name = var_name.strip()
            if var_name:
                # Case insensitive search
                mask = self.d_items_data[self.search_column].str.contains(var_name, case = False, na = False)
                matched_items = self.d_items_data[mask][self.id_column].tolist()
                results.extend(matched_items)
        
        return list(set(results))

    def search_by_keyword(self, keyword, 
                          output_column = None, use_regex = False, 
                          enable_translation = False, target_languages = None, 
                          translation_service = 'openai', api_key = None, custom_translator = None,
                          enable_interactive_mode = False):
        """
        Enhanced search with multilingual, case-insensitive support and interactive chatbot mode
        
        Parameters:
        - keyword: Original search keyword
        - output_column: Column to return from filtered results (default: use id_column)
        - use_regex: Whether to use regular expressions
        - enable_translation: Whether to enable translation
        - target_languages: List of target languages for translation (e.g., ['German', 'French'])
        - translation_service: 'openai', 'claude', or 'custom'
        - api_key: API key for translation service
        - custom_translator: Custom translation function
        - enable_interactive_mode: Whether to enable interactive chatbot mode if initial search is unsatisfactory
        """
        
        # Use id_column as default output_column if not specified
        if output_column is None:
            output_column = self.id_column
        
        # Store for potential interactive mode
        self.last_keyword = keyword
        self.last_api_service = translation_service
        self.last_api_key = api_key
        
        # Generate multiple case variations of the keyword for comprehensive search
        search_terms = set()  # Use set to avoid duplicates
        
        # Add original keyword
        search_terms.add(keyword)
        
        # Add various case variations
        search_terms.add(keyword.lower())           # all lowercase
        search_terms.add(keyword.upper())           # all uppercase
        search_terms.add(keyword.capitalize())      # first letter uppercase
        search_terms.add(keyword.title())           # title case (each word capitalized)
        
        # Convert back to list for processing
        search_terms = list(search_terms)
        
        # Add translated keywords if translation is enabled
        if enable_translation and target_languages:
            translated_terms = []
            for term in search_terms:
                for lang in target_languages:
                    if translation_service == 'openai' and api_key:
                        translated = self.translate_keyword_openai(term, lang, api_key)
                    elif translation_service == 'claude' and api_key:
                        translated = self.translate_keyword_claude(term, lang, api_key)
                    elif translation_service == 'custom' and custom_translator:
                        translated = self.translate_keyword_custom(term, lang, custom_translator)
                    else:
                        print(f"Translation service not configured properly for {lang}")
                        continue
                    
                    if translated and translated != term:
                        translated_terms.append(translated)
                        print(f"Translated '{term}' to {lang}: '{translated}'")
            
            search_terms.extend(translated_terms)
        
        # Remove duplicates from final search terms
        search_terms = list(set(search_terms))
        
        # Perform search with all terms (original + case variations + translated)
        all_filtered_data = pd.DataFrame()
        
        for term in search_terms:
            if use_regex:
                # Use regular expressions for searching with case insensitive flag
                pattern = re.compile(term, re.IGNORECASE)
                filtered_data = self.d_items_data[self.d_items_data[self.search_column].str.contains(pattern, na = False)]
            else:
                # Use simple string matching with case insensitive search
                filtered_data = self.d_items_data[self.d_items_data[self.search_column].str.contains(term, case = False, na = False)]
            
            all_filtered_data = pd.concat([all_filtered_data, filtered_data], ignore_index=True)
        
        # Remove duplicates and get results
        all_filtered_data = all_filtered_data.drop_duplicates()
        initial_results = all_filtered_data[output_column].tolist()
        
        # Store results for potential interactive mode
        self.last_search_results = initial_results
        
        # Check if interactive mode should be activated
        if enable_interactive_mode and api_key and translation_service in ['openai', 'claude']:
            print(f"\n🔍 Initial search found {len(initial_results)} results for '{keyword}'")
            
            if len(initial_results) == 0:
                print("❌ No results found with basic search.")
                activate_interactive = input("Would you like to try enhanced AI search? (y/n): ").lower()
            else:
                print(f"✅ Found {len(initial_results)} results")
                # Show all initial results
                if len(initial_results) > 0:
                    print("\n📋 All initial search results:")
                    for _, row in all_filtered_data.iterrows():
                        print(f"  - {row[self.search_column]} (ID: {row[self.id_column]})")
                
                activate_interactive = input("\nAre you satisfied with these results, or would you like to try enhanced AI search? (satisfied/enhance): ").lower()
                activate_interactive = 'y' if activate_interactive == 'enhance' else 'n'
            
            if activate_interactive == 'y':
                additional_results = self._interactive_chatbot_mode(keyword, translation_service, api_key)
                if additional_results:
                    # Combine initial results with additional results
                    combined_results = list(set(initial_results + additional_results))
                    new_items = [item for item in additional_results if item not in initial_results]
                    
                    print(f"\n🎉 FINAL RESULTS: {len(combined_results)} items total")
                    print(f"   📊 Initial search: {len(initial_results)} items")
                    print(f"   🆕 New items added: {len(new_items)} items")
                    print(f"   📈 Enhancement rate: {len(new_items)}/{len(initial_results)} = {len(new_items)/max(len(initial_results), 1):.1%}")
                    
                    # Show all final results with marking
                    print("\n📋 Complete final results list:")
                    initial_set = set(initial_results)
                    for item in combined_results:
                        if item in initial_set:
                            # Find the corresponding row in all_filtered_data
                            matching_row = all_filtered_data[all_filtered_data[self.id_column] == item]
                            if not matching_row.empty:
                                label = matching_row.iloc[0][self.search_column]
                                print(f"  ✅ {label} (ID: {item}) [Initial]")
                        else:
                            print(f"  🆕 (ID: {item}) [New - Enhanced Search]")
                    
                    return combined_results
        
        return initial_results
    
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
    
    def get_search_summary(self):
        """
        Get a summary of the last search performed
        """
        if self.last_search_results is None:
            return "No search has been performed yet."
        
        summary = f"""
        📊 Search Summary:
        - Keyword: '{self.last_keyword}'
        - Results found: {len(self.last_search_results)}
        - API service used: {self.last_api_service}
        """
        
        return summary
    
    def clear_search_history(self):
        """
        Clear the search history
        """
        self.last_search_results = None
        self.last_keyword = None
        self.last_api_service = None
        self.last_api_key = None
        print("🧹 Search history cleared.")

class VariableSelect:
    def __init__(self, d_items_table, item_id_list):
        self.d_items_data = d_items_table

        # Store 'item_id' lists
        self.variable_list = item_id_list

    def select_variables(self):
        self.d_items_data = self.d_items_data[self.d_items_data['itemid'].isin(self.variable_list)]

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

    def select_data_ventilationEvents(self, ventilation_id_list):
        self.d_items_data_ventilation = self.d_items_data[self.d_items_data['itemid'].isin(ventilation_id_list)].copy()
        return self.d_items_data_ventilation

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
    def __init__(self, ICU_patient_data, pro_events_data):
        self.ICU_patient_data = ICU_patient_data
        self.pro_events_data = pro_events_data

    def filter_ICU_patients(self, icu_unit_list):
        self.ICU_patient_data = self.ICU_patient_data[self.ICU_patient_data['first_careunit'].isin(icu_unit_list)]
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

    def calculate_los(self):
        self.ICU_patient_data['intime'] = pd.to_datetime(self.ICU_patient_data['intime'])
        self.ICU_patient_data['outtime'] = pd.to_datetime(self.ICU_patient_data['outtime'])
        self.ICU_patient_data['TD_LOS'] = (self.ICU_patient_data['outtime'] - self.ICU_patient_data['intime']).dt.days

    ### For extubation decision-making, we need to process the pro_events_data to get the ventilation events
    def process_pro_events_data(self, items_list_ventliation):
        self.pro_events_data = self.pro_events_data[self.pro_events_data['itemid'].isin(items_list_ventliation)]
        pro_events_vt = self.pro_events_data[self.pro_events_data['ordercategoryname'] == 'Ventilation'].copy()
        self.pro_events_data = self.pro_events_data[self.pro_events_data['stay_id'].isin(pro_events_vt['stay_id'])]
        self.pro_events_data = self.pro_events_data.reset_index(drop = True)

    def combine_ICU_patient_data_and_pro_events_data(self):
        self.ICU_patient_data = self.ICU_patient_data.sort_values(by = ['subject_id', 'intime'])
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

        self.ICU_patient_data = pd.merge(self.ICU_patient_data, self.pro_events_data, how = 'inner', on = 'stay_id')
        self.ICU_patient_data = self.ICU_patient_data.drop(columns = ['subject_id_y'])
        self.ICU_patient_data = self.ICU_patient_data.rename(columns = {'subject_id_x': 'subject_id'})
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

    ### Denote the Extubation Failure cases instead of readmission cases in discharge decision-making problem
    def denote_EF_cases(self, extubation_failure_days = 7):
        self.ICU_patient_data = self.ICU_patient_data.sort_values(by = ['subject_id', 'stay_id', 'starttime', 'endtime']).copy()
        self.ICU_patient_data['mv_id'] = 1
        for i in range(1, len(self.ICU_patient_data)):
            if self.ICU_patient_data['stay_id'].iloc[i] == self.ICU_patient_data['stay_id'].iloc[i-1]:
                self.ICU_patient_data['mv_id'].iloc[i] = self.ICU_patient_data['mv_id'].iloc[i-1] + 1
        
        admission_list = pd.unique(self.ICU_patient_data['stay_id'])

        ext_fail_list_day = []

        mv_id_record_day = []

        for i in tqdm(range(len(admission_list))):
            sub_table = self.ICU_patient_data[self.ICU_patient_data['stay_id'] == admission_list[i]]
            
            if len(sub_table) >= 2:
                for j in range(len(sub_table) - 1):
                    if sub_table['itemid'].iloc[j] == 225792: # this is a invasive MV event according to the d_items_data
                        for k in range(j + 1, len(sub_table)):
                            if sub_table['starttime'].iloc[k] - sub_table['endtime'].iloc[j] <= pd.Timedelta(f'{extubation_failure_days} days 00:00:00'):
                                ext_fail_list_day.append(admission_list[i])
                                mv_id_record_day.append(sub_table['mv_id'].iloc[j])
                                
                    else:
                        continue
            else:
                continue

        self.ICU_patient_data[f'ext_fail_{extubation_failure_days}_day'] = 0

        for i in range(len(self.ICU_patient_data)):
            for j in range(len(ext_fail_list_day)):
                if self.ICU_patient_data['stay_id'].iloc[i] == ext_fail_list_day[j]:
                    self.ICU_patient_data.loc[((self.ICU_patient_data['stay_id'] == ext_fail_list_day[j]) & (self.ICU_patient_data['mv_id'] == mv_id_record_day[j])), f'ext_fail_{extubation_failure_days}_day'] = 1

    ### The ICU extubation decision-making study needs additional data selection process compared with the discharge decision-making problem
    def data_selection_extubation(self):
        ### Exclude patients with over 30-day ICU LOS
        self.ICU_patient_data = self.ICU_patient_data[self.ICU_patient_data['los'] <= 30]
        
        ### Identify the non-invasive MV events
        self.NIV_list = []

        for i in range(len(self.ICU_patient_data)):
            if self.ICU_patient_data['itemid'].iloc[i] == 225794:
                self.NIV_list.append(self.ICU_patient_data['stay_id'].iloc[i])

        self.ICU_patient_data_NIV = self.ICU_patient_data[self.ICU_patient_data['itemid'] == 225794].copy()

        self.NIV_list_patient = pd.unique(self.ICU_patient_data_NIV['subject_id'])

        ### Identify unplanned extubation records and exclude them
        self.ICU_patient_data_up = self.ICU_patient_data[self.ICU_patient_data['itemid'].isin([225468, 225477])].copy()
        self.up_list_patient = pd.unique(self.ICU_patient_data_up['subject_id'])
        self.up_list_adm = pd.unique(self.ICU_patient_data_up['stay_id'])


        self.ICU_patient_data = self.ICU_patient_data[self.ICU_patient_data['itemid'].isin([224385, 225792, 225794, 227194])]
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)
        self.ICU_patient_data['intime'] = pd.to_datetime(self.ICU_patient_data['intime'])
        self.ICU_patient_data['outtime'] = pd.to_datetime(self.ICU_patient_data['outtime'])
        self.ICU_patient_data['starttime'] = pd.to_datetime(self.ICU_patient_data['starttime'])
        self.ICU_patient_data['endtime'] = pd.to_datetime(self.ICU_patient_data['endtime'])
        self.ICU_patient_data['TD_ICU'] = self.ICU_patient_data['outtime'] - self.ICU_patient_data['intime']
        self.ICU_patient_data['TD_MV'] = self.ICU_patient_data['endtime'] - self.ICU_patient_data['starttime']

        self.ICU_patient_data = self.ICU_patient_data[~self.ICU_patient_data['stay_id'].isin(self.up_list_adm)]

        ### Only consider the patient with a MV duration less than one week
        self.ICU_patient_data = self.ICU_patient_data[self.ICU_patient_data['TD_MV'] <= pd.Timedelta('7 days 00:00:00')]
        self.ICU_patient_data['RLOS'] = self.ICU_patient_data['outtime'] - self.ICU_patient_data['endtime']
        self.ICU_patient_data['LOS_initial'] = self.ICU_patient_data['outtime'] - self.ICU_patient_data['starttime']

        ### Exclude the readmitted cases within 30 days after discharging from the ICU, and we also need to keep their first admission record
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)
        self.ICU_patient_data = self.ICU_patient_data[self.ICU_patient_data['itemid'].isin([225792, 227194])]
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)
        pa_list = pd.unique(self.ICU_patient_data['subject_id'])
        ad_list = pd.unique(self.ICU_patient_data['stay_id'])
        # build the readmission list
        rd_list = []
        for i in range(len(pa_list)):
            sub_data = self.ICU_patient_data[self.ICU_patient_data['subject_id'] == pa_list[i]]
            if len(pd.unique(sub_data['stay_id'])) > 1:
                rd_list.append(pa_list[i])
        
        self.ICU_patient_data_rd = self.ICU_patient_data[self.ICU_patient_data['subject_id'].isin(rd_list)].copy()
        pa_list_2 = []
        drop_list_2 = []

        for i in range(len(rd_list)):
            sub_data = self.ICU_patient_data_rd[self.ICU_patient_data_rd['subject_id'] == rd_list[i]]
            for j in range(1, len(sub_data)):
                if sub_data['stay_id'].iloc[j] != sub_data['stay_id'].iloc[j-1]:
                    if sub_data['intime'].iloc[j] - sub_data['outtime'].iloc[j-1] <= pd.Timedelta('30 days 00:00:00'):
                        pa_list_2.append(rd_list[i])
                        drop_list_2.append(sub_data['stay_id'].iloc[j - 1])
                        break
                else:
                    if sub_data['stay_id'].iloc[j] != sub_data['stay_id'].iloc[j-1]:
                        if sub_data['intime'].iloc[j] - sub_data['outtime'].iloc[j-1] <= pd.Timedelta('30 days 00:00:00'):
                            pa_list_2.append(rd_list[i])                    
                            drop_list_2.extend(pd.unique(sub_data['stay_id'])[1:])
                            break

        self.ICU_patient_data = self.ICU_patient_data[~self.ICU_patient_data['stay_id'].isin(drop_list_2)].copy()
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

        ### Exclude the patients with an NIV record during their invaisve MV treatment because the exact length of invasive MV could not be determined
        admission_list = pd.unique(self.ICU_patient_data['stay_id'])
        drop_list_1 = []
        drop_list_2 = []
        drop_list_mv = []
        for i in tqdm(range(len(admission_list))):
            sub_table = self.ICU_patient_data[self.ICU_patient_data['stay_id'] == admission_list[i]]
            
            if len(pd.unique(sub_table['itemid'])) == 1:
                if pd.unique(sub_table['itemid']) == 225794:
                    drop_list_1.append(admission_list[i])
                else:
                    continue
            else:
                for j in range(len(sub_table) - 1):
                    if sub_table['itemid'].iloc[j] == 225794:
                        for k in range(j + 1, len(sub_table)):
                            if sub_table['itemid'].iloc[k] == 225792:
                                if sub_table['starttime'].iloc[k] - sub_table['endtime'].iloc[j] < pd.Timedelta('0 days 00:00:00'):
                                    drop_list_2.append(admission_list[i])
                                    drop_list_mv.append(sub_table['mv_id'].iloc[j])
                    else:
                        for k in range(j + 1, len(sub_table)):
                            if sub_table['itemid'].iloc[k] == 225794:
                                if sub_table['starttime'].iloc[k] - sub_table['endtime'].iloc[j] < pd.Timedelta('0 days 00:00:00'):
                                    drop_list_2.append(admission_list[i])
                                    drop_list_mv.append(sub_table['mv_id'].iloc[j])
        
        self.ICU_patient_data = self.ICU_patient_data[~self.ICU_patient_data['stay_id'].isin(drop_list_1)]
        for i in range(len(drop_list_2)):
            condition = (self.ICU_patient_data['stay_id'] == drop_list_2[i]) & (self.ICU_patient_data['mv_id'] == drop_list_mv[i])
            self.ICU_patient_data = self.ICU_patient_data[~condition]
        
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)


    ### We still need to denote the death cases in the ICU patient data for the extubation decision-making problem
    ### But we now care about the death time after the extubation, not the death time after the ICU discharge
    def denote_death_cases(self, admission_data, patients_data, death_observation_days = 30):

        patients_data_select = patients_data.drop(columns = ['anchor_year'])
        admission_data_select = admission_data[['subject_id', 'hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type', 'insurance', 'race']]

        patients_data_select = patients_data_select[patients_data_select['subject_id'].isin(self.ICU_patient_data['subject_id'])]
        admission_data_select = admission_data_select[admission_data_select['subject_id'].isin(self.ICU_patient_data['subject_id'])]

        admission_data_select_v1 = admission_data_select[['subject_id', 'race']].copy()
        admission_data_select_v1 = admission_data_select_v1.drop_duplicates(subset = ['subject_id'], keep = 'first')

        self.ICU_patient_data = pd.merge(self.ICU_patient_data, admission_data_select_v1, how = 'left', on = 'subject_id')
        self.ICU_patient_data = pd.merge(self.ICU_patient_data, patients_data_select, how = 'left', on = 'subject_id')
        
        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

        self.ICU_patient_data['dod'] = pd.to_datetime(self.ICU_patient_data['dod'])
        self.ICU_patient_data['TD_death_ext'] = self.ICU_patient_data['dod'] - self.ICU_patient_data['endtime']

        self.ICU_patient_data['death_in_MV'] = 0
        self.ICU_patient_data[f'death_after_MV_{death_observation_days}_day'] = 0

        self.ICU_patient_data.loc[self.ICU_patient_data['TD_death_ext'] <= pd.Timedelta(0), 'death_in_MV'] = 1
        self.ICU_patient_data.loc[(self.ICU_patient_data['TD_death_ext'] > pd.Timedelta(0)) & 
                                  (self.ICU_patient_data['TD_death_ext'] <= pd.Timedelta(days = death_observation_days)), f'death_after_MV_{death_observation_days}_day'] = 1

        self.ICU_patient_data = self.ICU_patient_data.reset_index(drop = True)

    ### We now need to denote the MV event count instead of readmission count in the extubation decision-making problem
    def denote_ext_time(self, extubation_failure_days = 7):
        self.ICU_patient_data = self.ICU_patient_data.sort_values(by = ['subject_id', 'stay_id', 'starttime', 'endtime'])
        self.ICU_patient_data['ext_time'] = 1

        for i in range(1, len(self.ICU_patient_data)):
            if self.ICU_patient_data['stay_id'].iloc[i] == self.ICU_patient_data['stay_id'].iloc[i-1]:
                if self.ICU_patient_data[f'ext_fail_{extubation_failure_days}_day'].iloc[i-1] == 1:
                    self.ICU_patient_data['ext_time'].iloc[i] = self.ICU_patient_data['ext_time'].iloc[i-1] + 1
                elif self.ICU_patient_data['mv_id'].iloc[i] != self.ICU_patient_data['mv_id'].iloc[i-1]:
                    self.ICU_patient_data['ext_time'].iloc[i] = self.ICU_patient_data['ext_time'].iloc[i-1] + 1
                            

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

        self.icu_patient_data = self.icu_patient_data[self.icu_patient_data['stay_id'].isin(self.chart_events_data['stay_id'])]
        self.icu_patient_data = self.icu_patient_data.reset_index(drop = True)

        self.chart_events_data['charttime'] = pd.to_datetime(self.chart_events_data['charttime'])
        self.chart_events_data['storetime'] = pd.to_datetime(self.chart_events_data['storetime'])
    
    def data_selection(self, data, i_1, i_2, i_3):
        sub_data = data.loc[(data['charttime'] >= i_1) & (data['charttime'] <= i_2) & (data["itemid"] == i_3)]
        return sub_data

    ### physio_table needs to be constructed by the same way as the discharge decision-making problem
    def dataset_generation(self, physio_table, extubation_failure_days = 7, death_observation_days = 30):
        ### different from the discharge decision-making problem, we need to go through the stay_id with multiple MV events
        icu_stay_list = list(self.icu_patient_data['stay_id'])

        for i in range(len(icu_stay_list)):
            
            print("The number of processed ICU stay admissions with IMV treatment: ", i)
            
            index = self.icu_patient_data["starttime"].iloc[i]
            
            s_table_id = self.chart_events_data[self.chart_events_data['stay_id'] == icu_stay_list[i]]

            while index <= self.icu_patient_data["endtime"].iloc[i]:
                physio_table['subject_id'].append(self.icu_patient_data['subject_id'].iloc[i])
                physio_table['hadm_id'].append(self.icu_patient_data['hadm_id'].iloc[i])
                physio_table['stay_id'].append(self.icu_patient_data['stay_id'].iloc[i])
                physio_table['icu_starttime'].append(self.icu_patient_data['intime'].iloc[i])
                physio_table['icu_endtime'].append(self.icu_patient_data['outtime'].iloc[i])
                physio_table['IMV_starttime'].append(self.icu_patient_data['starttime'].iloc[i])
                physio_table['IMV_endtime'].append(self.icu_patient_data['endtime'].iloc[i])
                physio_table['Total_LOS'].append(self.icu_patient_data['los'].iloc[i])
                physio_table['RLOS'].append(self.icu_patient_data['RLOS'].iloc[i])
                physio_table['LOS_initial'].append(self.icu_patient_data['LOS_initial'].iloc[i])
                physio_table['ext_fail'].append(self.icu_patient_data[f'ext_fail_{extubation_failure_days}_day'].iloc[i])
                physio_table['ext_count'].append(self.icu_patient_data['ext_time'].iloc[i])
                physio_table['insurance'].append(self.icu_patient_data['insurance'].iloc[i])
                physio_table['death_in_MV'].append(self.icu_patient_data['death_in_MV'].iloc[i])
                physio_table['death_after_MV'].append(self.icu_patient_data[f'death_after_MV_{death_observation_days}_day'].iloc[i])
                physio_table['year_group'].append(self.icu_patient_data['anchor_year_group'].iloc[i])
                physio_table['age'].append(self.icu_patient_data['anchor_age'].iloc[i])
                physio_table['gender'].append(self.icu_patient_data['gender'].iloc[i])
                physio_table['race'].append(self.icu_patient_data['race'].iloc[i])
                physio_table['patientweight'].append(self.icu_patient_data['patientweight'].iloc[i])
                
                td = pd.Timedelta('0 days 06:00:00')
                
                index_1 = index + td

                if index_1 <= self.icu_patient_data["endtime"].iloc[i]:
                    physio_table['time'].append(index_1)
                else:
                    index_1 = self.icu_patient_data["endtime"].iloc[i]
                    physio_table['time'].append(index_1)

                for j in range(len(self.d_items_data_chart)):
                    s_table = self.data_selection(s_table_id, 
                                        index, 
                                        index_1,
                                        self.d_items_data_chart["itemid"].iloc[j])

                    n = len(s_table)

                    if n >= 1:
                        physio_table[self.d_items_data_chart['label'].iloc[j]].append(s_table['valuenum'].mean())

                    else:
                        physio_table[self.d_items_data_chart['label'].iloc[j]].append(np.nan)

                index = index + td
        
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

    def process_gen_data(self, drop_columns = []):
        ### The unit of the Tidal Volume is ml, we need to convert it to L
        ### It is easier for following computation of RSBI
        self.generated_dataset['Tidal Volume (set)'] = self.generated_dataset['Tidal Volume (set)']/1000
        self.generated_dataset['Tidal Volume (observed)'] = self.generated_dataset['Tidal Volume (observed)']/1000
        self.generated_dataset['Tidal Volume (spontaneous)'] = self.generated_dataset['Tidal Volume (spontaneous)']/1000

        gender_dummies = pd.get_dummies(self.generated_dataset['gender'])
        self.generated_dataset = pd.concat([self.generated_dataset, gender_dummies], axis = 'columns')
        self.generated_dataset = self.generated_dataset.drop(['gender', 'F'], axis = 'columns')

        self.generated_dataset = self.generated_dataset.drop(columns = ['race'])

        icu_stayid_list = list(self.icu_patient_data['stay_id'])
        icu_mv_count_list = list(self.icu_patient_data['ext_count'])

        ### Extubation action instead of discharge action in the extubation decision-making problem
        self.generated_dataset['ext_action'] = 0

        for i in range(len(icu_stayid_list)):
            time_idx = self.generated_dataset[(self.generated_dataset['stay_id'] == icu_stayid_list[i]) & (self.generated_dataset['ext_count'] == icu_mv_count_list[i])]['time'].iloc[-1]
            self.generated_dataset.loc[(self.generated_dataset['stay_id'] == icu_stayid_list[i]) & (self.generated_dataset['ext_count'] == icu_mv_count_list[i]) & (self.generated_dataset['time'] == time_idx), 'ext_action'] = 1

        self.generated_dataset['Blood Pressure Systolic'] = self.generated_dataset.apply(self.assign_blood_pressure, axis = 1)
        self.generated_dataset['Blood Pressure Diastolic'] = self.generated_dataset.apply(self.assign_blood_pressure_diastolic, axis = 1)
        self.generated_dataset['Blood Pressure Mean'] = self.generated_dataset.apply(self.assign_blood_pressure_mean, axis = 1)

        self.generated_dataset['Temperature C'] = self.generated_dataset.apply(self.assign_temperature, axis = 1)
        self.generated_dataset['SaO2'] = self.generated_dataset.apply(self.assign_SaO2, axis = 1)

        self.generated_dataset['GCS Score'] = self.generated_dataset.apply(self.assign_gcs_score, axis = 1)
        self.generated_dataset['PEEP Level'] = self.generated_dataset.apply(self.assign_peep_level, axis = 1)
        self.generated_dataset['Weight'] = self.generated_dataset.apply(self.assign_weight, axis = 1)

        self.generated_dataset = self.generated_dataset.drop(columns = drop_columns)

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

    def classify_missing_columns(self, var_list = [], missing_threshold_1 = 0.75, missing_threshold_2 = 0.10):
        drop_list = []
        middle_list = []
        knn_list = []
        for i in var_list:
            if (self.generated_dataset[i].isnull().sum()/len(self.generated_dataset)) > missing_threshold_1:
                drop_list.append(i)
            elif ((self.generated_dataset[i].isnull().sum()/len(self.generated_dataset)) <= missing_threshold_1) and ((self.generated_dataset[i].isnull().sum()/len(self.generated_dataset)) >= missing_threshold_2):
                middle_list.append(i)
            else:
                knn_list.append(i)
        
        return drop_list, middle_list, knn_list

    def forward_fill_missing_values(self, according_list = [], var_list = []):
        for i in range(len(var_list)):
            self.generated_dataset[var_list[i]] = self.generated_dataset.groupby(by = according_list)[var_list[i]].ffill()

    ### note that select the variables suitable for mean imputation
    def mean_impute_missing_values(self, according_list = [], var_list = []):
        for var in var_list:
            self.generated_dataset[var] = (self.generated_dataset.groupby(by = according_list)[var].transform(lambda s: s.fillna(s.mean())))

    ### note that select the variables suitable for linear interpolation
    def linear_impute_missing_values(self, according_list = [], var_list = []):
        for i in range(len(var_list)):
            self.generated_dataset[var_list[i]] = self.generated_dataset.groupby(by = according_list)[var_list[i]].apply(lambda x: x.interpolate(method = 'linear'))

    def process_chunk(self, chunk, imputer):
        chunk_imputed = imputer.fit_transform(chunk)  
        return chunk_imputed

    def knn_impute_missing_values(self, num_neigh = 5, 
                                  scaler = MinMaxScaler(), 
                                  chunk_size = 10000, num_jobs = 60):

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
        self.scaler = None  # Store the fitted scaler
        self.scaler_feature_names = None  # Store the feature names used for scaling

    def drop_duplicate_rows(self, subset = []):
        ### note that subset is the columns to be considered for dropping duplicates
        self.generated_dataset = self.generated_dataset.drop_duplicates(subset = subset, keep = 'first')
        
        m = self.generated_dataset[self.generated_dataset['ext_action'] == 1]
        duplicates = m[m.duplicated(subset = subset)]
        if len(duplicates) > 0:
            print(f"Attention!!! Found {len(duplicates)} duplicate rows in the dataset! Please check the dataset manually.")
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

    ### Denote the decision epoch in the dataset
    def denote_decision_epoch(self):
        self.generated_dataset['epoch'] = self.generated_dataset.groupby(['stay_id', 'ext_count']).cumcount() + 1
        
    def additional_icu_extubation_data_selection(self):

        self.generated_dataset = self.generated_dataset[self.generated_dataset['death_in_MV'] != 1].copy()
        self.generated_dataset = self.generated_dataset[self.generated_dataset['death_after_MV'] != 1].copy()
        
        self.generated_dataset = self.generated_dataset.reset_index(drop = True)

    def denote_MV_event(self):
        self.generated_dataset['mv_event_id'] = 1

        for i in range(1, len(self.generated_dataset)):
            if (self.generated_dataset['stay_id'].iloc[i] != self.generated_dataset['stay_id'].iloc[i-1]) | (self.generated_dataset['ext_count'].iloc[i] != self.generated_dataset['ext_count'].iloc[i-1]):
                self.generated_dataset['mv_event_id'].iloc[i] = self.generated_dataset['mv_event_id'].iloc[i-1] + 1
            else:
                self.generated_dataset['mv_event_id'].iloc[i] = self.generated_dataset['mv_event_id'].iloc[i-1]

    def denote_qsofa(self):
        self.generated_dataset['qsofa'] = self.generated_dataset.apply(self.compute_qsofa, axis = 1)

    def denote_rsbi(self):
        self.generated_dataset['rsbi'] = self.generated_dataset.apply(self.compute_rsbi, axis = 1)

    def table_split(self, var_outcome = [], var_physio = []):
        self.rl_cont_state_table = self.generated_dataset[var_physio].copy()
        self.state_id_table = self.generated_dataset[var_outcome].copy()

    ### Denote the extubation cost instead of the discharge cost
    def extubation_cost_set(self, scaler = MinMaxScaler()):
        # extubation failure risk costs
        self.state_id_table['extubation_fail_costs'] = 0

        condition_1 = (self.state_id_table['ext_action'] == 1) & (self.state_id_table['ext_fail'] == 1)
        self.state_id_table.loc[condition_1, 'extubation_fail_costs'] = 1

        condition_2 = (self.state_id_table['ext_action'] == 1) & (self.state_id_table['ext_fail'] != 1)
        self.state_id_table.loc[condition_2, 'extubation_fail_costs'] = 0

        # length-of-stay costs - LOS in the ICU after the initiation of invasive MV treatment
        self.state_id_table['time'] = pd.to_datetime(self.state_id_table['time'])
        self.state_id_table['icu_starttime'] = pd.to_datetime(self.state_id_table['icu_starttime'])
        self.state_id_table['icu_endtime'] = pd.to_datetime(self.state_id_table['icu_endtime'])
        self.state_id_table['IMV_starttime'] = pd.to_datetime(self.state_id_table['IMV_starttime'])
        self.state_id_table['IMV_endtime'] = pd.to_datetime(self.state_id_table['IMV_endtime'])
        self.state_id_table['RLOS'] = self.state_id_table['RLOS']/pd.Timedelta('1 hour')
        self.state_id_table['LOS_initial'] = self.state_id_table['LOS_initial']/pd.Timedelta('1 hour')

        self.state_id_table['los_costs'] = 0.0

        ext_action_zero_mask = self.state_id_table['ext_action'] == 0
        self.state_id_table.loc[ext_action_zero_mask, 'los_costs'] = 6.0

        ext_action_one_mask = self.state_id_table['ext_action'] == 1
        self.state_id_table.loc[ext_action_one_mask, 'los_costs'] = self.state_id_table.loc[ext_action_one_mask, 'RLOS']

        ### It needs addition process for the RLOS
        self.state_id_table['rlos_start_check'] = self.state_id_table['icu_endtime'] - self.state_id_table['IMV_starttime']

        drop_mv_id_list = []

        for i in range(len(self.state_id_table)):
            if self.state_id_table['rlos_start_check'].iloc[i] <= pd.Timedelta('0 days 00:00:00'):
                drop_mv_id_list.append(self.state_id_table['mv_event_id'].iloc[i])

        self.state_id_table = self.state_id_table[~self.state_id_table['mv_event_id'].isin(drop_mv_id_list)].copy()
        self.state_id_table = self.state_id_table.reset_index(drop = True)

        drop_mv_id_list = []

        for i in range(len(self.state_id_table)):
            if self.state_id_table['RLOS'].iloc[i] <= -0.5:
                drop_mv_id_list.append(self.state_id_table['mv_event_id'].iloc[i])
                
        self.state_id_table = self.state_id_table[~self.state_id_table['mv_event_id'].isin(drop_mv_id_list)].copy()
        self.state_id_table = self.state_id_table.reset_index(drop = True)

        condition = (self.state_id_table['RLOS'] <= 0)
        self.state_id_table.loc[condition, 'RLOS'] = 0

        self.state_id_table['los_costs'] = 0.0

        ext_action_zero_mask = self.state_id_table['ext_action'] == 0
        self.state_id_table.loc[ext_action_zero_mask, 'los_costs'] = 6.0

        ext_action_one_mask = self.state_id_table['ext_action'] == 1
        self.state_id_table.loc[ext_action_one_mask, 'los_costs'] = self.state_id_table.loc[ext_action_one_mask, 'RLOS']

        mv_id_list = self.state_id_table['mv_event_id'].unique()
        self.generated_dataset = self.generated_dataset[self.generated_dataset['mv_event_id'].isin(mv_id_list)].copy()
        ### please do the table split again ###

        self.state_id_table['los_costs_scaled'] = 0
        self.state_id_table[['los_costs_scaled']] = scaler.fit_transform(self.state_id_table[['los_costs']])

        self.state_id_table = self.state_id_table.reset_index(drop = True)

    def qSOFA_safe_action_space(self):
        safe_condition = (self.state_id_table['qSOFA'] == 0) | (self.state_id_table['qSOFA'] == 1)
        unsafe_condition = (self.state_id_table['qSOFA'] == 2) | (self.state_id_table['qSOFA'] == 3)

        self.state_id_table.loc[safe_condition, 'qsofa_safe_action'] = 1.0
        self.state_id_table.loc[unsafe_condition, 'qsofa_safe_action'] = 0.0

    ### new clinical safe action indicator based on the RSBI score
    def rsbi_safe_action_space(self):
        safe_condition = (self.state_id_table['rsbi'] <= 105) & (self.state_id_table['rsbi'] > 0)
        unsafe_condition = (self.state_id_table['rsbi'] > 105) | (self.state_id_table['rsbi'] == 0)

        self.state_id_table.loc[safe_condition, 'rsbi_safe_action'] = 1.0
        self.state_id_table.loc[unsafe_condition, 'rsbi_safe_action'] = 0.0

    def train_val_test_split(self, scaler = MinMaxScaler(), test_prop = 0.2, val_prop = 0.5, random_seed = 42):
        self.rl_cont_state_table.rename(columns = {'RR': 'Respiratory Rate', 'TV': 'Tidal Volume'}, inplace = True)
        self.rl_cont_state_table['age'] = self.rl_cont_state_table['age'].astype(float)
        self.rl_cont_state_table['M'] = self.rl_cont_state_table['M'].astype(float)

        self.state_id_table['ext_action'] = self.state_id_table['ext_action'].astype(float)
        self.state_id_table['ext_fail'] = self.state_id_table['ext_fail'].astype(float)
        self.state_id_table['extubation_fail_costs'] = self.state_id_table['extubation_fail_costs'].astype(float)
        self.state_id_table['los_costs'] = self.state_id_table['los_costs'].astype(float)
        self.state_id_table['los_costs_scaled'] = self.state_id_table['los_costs_scaled'].astype(float)

        var_list = self.rl_cont_state_table.columns.tolist()
        self.rl_cont_state_table_scaled = self.rl_cont_state_table.copy()
        
        # Store the scaler and feature names for later use
        self.scaler = scaler
        self.scaler_feature_names = var_list
        
        # Fit and transform the data
        self.rl_cont_state_table_scaled[var_list] = self.scaler.fit_transform(self.rl_cont_state_table[var_list])
        self.rl_cont_state_table_scaled['M'] = self.rl_cont_state_table['M'].copy()

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

    def compute_rsbi(self, row):
        score = np.nan

        if row['TV'] != 0:
            score = row['RR']/row['TV']

        return score
    
    def save_scaler(self, file_path, save_feature_names = True):
        """
        Save the fitted scaler to a file for later use
        
        Parameters:
        - file_path: Path to save the scaler (should end with .pkl)
        - save_feature_names: Whether to save feature names (default: True)
                              If False, only saves the scaler object for simpler usage
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet. Please run train_val_test_split first.")
        
        if save_feature_names:
            scaler_data = {
                'scaler': self.scaler,
                'feature_names': self.scaler_feature_names
            }
        else:
            # Simple version: only save the scaler object
            scaler_data = self.scaler
        
        with open(file_path, 'wb') as f:
            pickle.dump(scaler_data, f)
        
        if save_feature_names:
            print(f"Scaler with feature names saved to {file_path}")
        else:
            print(f"Scaler (simplified) saved to {file_path}")
    
    def load_scaler(self, file_path, has_feature_names = None):
        """
        Load a previously saved scaler
        
        Parameters:
        - file_path: Path to the saved scaler file
        - has_feature_names: None (auto-detect), True, or False
        """
        with open(file_path, 'rb') as f:
            scaler_data = pickle.load(f)
        
        # Auto-detect format if not specified
        if has_feature_names is None:
            if isinstance(scaler_data, dict):
                has_feature_names = True
            else:
                has_feature_names = False
        
        if has_feature_names:
            self.scaler = scaler_data['scaler']
            self.scaler_feature_names = scaler_data['feature_names']
            print(f"Scaler loaded from {file_path}")
            print(f"Feature names: {self.scaler_feature_names}")
        else:
            self.scaler = scaler_data
            self.scaler_feature_names = None
            print(f"Scaler (simplified) loaded from {file_path}")
            print("Note: No feature names available - ensure data order consistency")
    
    def get_scaler(self):
        """
        Get the fitted scaler object
        
        Returns:
        - scaler: The fitted MinMaxScaler object
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet. Please run train_val_test_split first.")
        return self.scaler
    
    def get_scaler_params(self):
        """
        Get the scaler parameters (min_, scale_, etc.)
        
        Returns:
        - dict: Dictionary containing scaler parameters and optionally feature names
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet. Please run train_val_test_split first.")
        
        params = {
            'min_': self.scaler.min_,
            'scale_': self.scaler.scale_,
            'data_min_': self.scaler.data_min_,
            'data_max_': self.scaler.data_max_,
            'data_range_': self.scaler.data_range_,
        }
        
        # Add feature names if available
        if self.scaler_feature_names is not None:
            params['feature_names'] = self.scaler_feature_names
        
        return params
    
    def transform_new_data(self, data, validate_features = True):
        """
        Transform new data using the fitted scaler
        
        Parameters:
        - data: DataFrame with the same features as training data
        - validate_features: Whether to validate feature names (default: True)
                           Set to False for simplified usage without feature name checking
        
        Returns:
        - transformed_data: DataFrame with scaled features
        """
        if self.scaler is None:
            raise ValueError("Scaler has not been fitted yet. Please run train_val_test_split first.")
        
        # Create a copy of the data
        transformed_data = data.copy()
        
        if validate_features and self.scaler_feature_names is not None:
            # Check if all required features are present
            missing_features = set(self.scaler_feature_names) - set(data.columns)
            if missing_features:
                raise ValueError(f"Missing features in input data: {missing_features}")
            
            # Transform only the features that were used in training
            transformed_data[self.scaler_feature_names] = self.scaler.transform(data[self.scaler_feature_names])
        else:
            # Simplified mode: transform all numeric columns
            if self.scaler_feature_names is not None:
                # Use saved feature names if available
                transformed_data[self.scaler_feature_names] = self.scaler.transform(data[self.scaler_feature_names])
            else:
                # No feature names saved, assume user provides data in correct order
                numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_columns) != len(self.scaler.min_):
                    print(f"Warning: Expected {len(self.scaler.min_)} features, got {len(numeric_columns)}")
                    print("Ensure data is in the same order as training data")
                
                transformed_data[numeric_columns] = self.scaler.transform(data[numeric_columns])
        
        return transformed_data
    
    def save_to_csv(self, dataset, file_path):
        dataset.to_csv(file_path, index = False)