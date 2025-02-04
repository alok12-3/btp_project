# # import streamlit as st
# # import math
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # # --- Title ---
# # st.title("Fluid Flow and Heat Transfer Calculator")
# #
# # # --- Introduction ---
# # st.write("""
# # This application calculates various fluid flow and heat transfer parameters such as pressure drop, Reynolds number, friction factor, Nusselt number, and TPF (Thermal Performance Factor).
# #
# # You can also visualize the effect of varying certain parameters on these quantities by generating plots.
# # """)
# #
# # # --- Input Section ---
# # st.header("Input Parameters")
# #
# # # Define default values
# # default_values = {
# #     'P_outlet': 101325,  # Pa
# #     'P_inlet': 101325,  # Pa
# #     'D_h': 0.01,  # m
# #     'rho': 1000,  # kg/m³
# #     'U': 1.0,  # m/s
# #     'L': 1.0,  # m
# #     'mu': 0.001,  # Pa.s
# #     'Pr': 7.0,  # Prandtl number for water at room temp
# #     'h': 500,  # W/m²K
# #     'k': 0.6  # W/mK
# # }
# #
# # col1, col2 = st.columns(2)
# # with col1:
# #     P_outlet = st.number_input("Outlet pressure (P_outlet) in Pascals", value=default_values['P_outlet'])
# #     P_inlet = st.number_input("Inlet pressure (P_inlet) in Pascals", value=default_values['P_inlet'])
# #     D_h = st.number_input("Hydraulic diameter (D_h) in meters", value=default_values['D_h'])
# #     rho = st.number_input("Fluid density (rho) in kg/m³", value=default_values['rho'])
# #     U = st.number_input("Mean velocity (U) in m/s", value=default_values['U'])
# #
# # with col2:
# #     L = st.number_input("Channel length (L) in meters", value=default_values['L'])
# #     mu = st.number_input("Dynamic viscosity (mu) in Pa.s", value=default_values['mu'])
# #     Pr = st.number_input("Prandtl number (Pr)", value=default_values['Pr'])
# #     h = st.number_input("Surface heat transfer coefficient (h) in W/m²K", value=default_values['h'])
# #     k = st.number_input("Thermal conductivity (k) in W/mK", value=default_values['k'])
# #
# # # --- Calculations ---
# # delta_p = P_outlet - P_inlet
# # Re = (rho * U * D_h) / mu
# # if Re <= 0:
# #     st.error("Invalid Reynolds number. Please check your inputs.")
# #     st.stop()
# # try:
# #     f_o_smooth = 1 / (1.82 * math.log10(Re) - 1.64) ** 2
# # except ValueError:
# #     st.error("Calculation of smooth friction factor failed due to invalid Reynolds number.")
# #     st.stop()
# # f = (2 * delta_p * D_h) / (rho * U ** 2 * L)
# # Nu_o_smooth = ((f_o_smooth / 8) * (Re - 1000) * Pr) / (1 + 12.7 * math.sqrt(f_o_smooth / 8) * (Pr ** (2 / 3) - 1))
# # Nu = (h * L) / k
# # TPF = (Nu / Nu_o_smooth) / ((f / f_o_smooth) ** (1 / 3))
# #
# # # --- Output ---
# # st.header("Calculated Parameters")
# # st.write(f"**Pressure drop (delta_p):** {delta_p:.2f} Pa")
# # st.write(f"**Reynolds number (Re):** {Re:.2f}")
# # st.write(f"**Friction factor (f):** {f:.5f}")
# # st.write(f"**Smooth friction factor (f_o,smooth):** {f_o_smooth:.5f}")
# # st.write(f"**Smooth Nusselt number (Nu_o,smooth):** {Nu_o_smooth:.2f}")
# # st.write(f"**Nusselt number (Nu):** {Nu:.2f}")
# # st.write(f"**TPF (Thermal Performance Factor):** {TPF:.4f}")
# #
# # # --- Plotting Section ---
# # st.header("Visualization")
# # st.write("""
# # Select the variable you want to vary and the quantity you want to plot against it.
# # """)
# #
# # # Variables that can be varied
# # variable_options = {
# #     'U (Mean velocity)': 'U',
# #     'Reynolds number (Re)': 'Re',
# #     'Surface heat transfer coefficient (h)': 'h',
# # }
# #
# # # Quantities that can be plotted
# # quantity_options = {
# #     'Nusselt number (Nu)': 'Nu',
# #     'TPF (Thermal Performance Factor)': 'TPF',
# #     'Friction factor (f)': 'f',
# # }
# #
# # # Select variable to vary
# # var_to_vary = st.selectbox("Select the variable to vary", list(variable_options.keys()))
# # var_name = variable_options[var_to_vary]
# #
# # # Select quantity to plot
# # quantity_to_plot = st.selectbox("Select the quantity to plot", list(quantity_options.keys()))
# # quantity_name = quantity_options[quantity_to_plot]
# #
# # # Get range for variable
# # st.subheader(f"Set range for {var_to_vary}")
# # var_min = st.number_input(f"Minimum {var_to_vary}", value=U if var_name == 'U' else (0.1 if var_name == 'h' else Re))
# # var_max = st.number_input(f"Maximum {var_to_vary}",
# #                           value=U * 10 if var_name == 'U' else (1000 if var_name == 'h' else Re * 10))
# # var_steps = st.number_input(f"Number of steps", min_value=2, value=50, step=1)
# #
# # if var_min >= var_max:
# #     st.error("Minimum value must be less than maximum value.")
# #     st.stop()
# #
# # # Compute over the range
# # var_values = np.linspace(var_min, var_max, int(var_steps))
# # quantity_values = []
# #
# # # Keep other variables constant
# # constant_vars = {
# #     'P_outlet': P_outlet,
# #     'P_inlet': P_inlet,
# #     'D_h': D_h,
# #     'rho': rho,
# #     'U': U,
# #     'L': L,
# #     'mu': mu,
# #     'Pr': Pr,
# #     'h': h,
# #     'k': k,
# # }
# #
# #
# # # Function to compute desired quantity
# # def compute_quantity(var_value):
# #     vars = constant_vars.copy()
# #     vars[var_name] = var_value
# #
# #     delta_p = vars['P_outlet'] - vars['P_inlet']
# #     Re = (vars['rho'] * vars['U'] * vars['D_h']) / vars['mu']
# #     if Re <= 0:
# #         return np.nan
# #     try:
# #         f_o_smooth = 1 / (1.82 * math.log10(Re) - 1.64) ** 2
# #         Nu_o_smooth = ((f_o_smooth / 8) * (Re - 1000) * vars['Pr']) / (
# #                     1 + 12.7 * math.sqrt(f_o_smooth / 8) * (vars['Pr'] ** (2 / 3) - 1))
# #         f = (2 * delta_p * vars['D_h']) / (vars['rho'] * vars['U'] ** 2 * vars['L'])
# #         Nu = (vars['h'] * vars['L']) / vars['k']
# #         TPF = (Nu / Nu_o_smooth) / ((f / f_o_smooth) ** (1 / 3))
# #     except (ValueError, ZeroDivisionError):
# #         return np.nan
# #
# #     quantities = {
# #         'Nu': Nu,
# #         'TPF': TPF,
# #         'f': f,
# #         'Re': Re
# #     }
# #
# #     return quantities[quantity_name]
# #
# #
# # # Compute quantities
# # for v in var_values:
# #     q = compute_quantity(v)
# #     quantity_values.append(q)
# #
# # # Remove NaN values
# # var_values = np.array(var_values)
# # quantity_values = np.array(quantity_values)
# # mask = ~np.isnan(quantity_values)
# # var_values = var_values[mask]
# # quantity_values = quantity_values[mask]
# #
# # # Plotting
# # fig, ax = plt.subplots()
# # ax.plot(var_values, quantity_values, marker='o')
# # ax.set_xlabel(f"{var_to_vary}")
# # ax.set_ylabel(f"{quantity_to_plot}")
# # ax.set_title(f"{quantity_to_plot} vs {var_to_vary}")
# # ax.grid(True)
# # st.pyplot(fig)
#
#
# import streamlit as st
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # --- Title ---
# st.title("Fluid Flow and Heat Transfer Calculator with Data Logging")
#
# # --- Introduction ---
# st.write("""
# This application calculates various fluid flow and heat transfer parameters such as pressure drop, Reynolds number, friction factor, Nusselt number, and TPF (Thermal Performance Factor).
#
# Each calculation is logged into a table, which you can download as a CSV file. You can also plot any two parameters from the table, with the option to exclude specific rows.
# """)
#
# # --- Initialize Session State ---
# if 'data' not in st.session_state:
#     st.session_state['data'] = pd.DataFrame()
#
# # --- Input Section ---
# st.header("Input Parameters")
#
# # Define default values
# default_values = {
#     'P_outlet': 201325,    # Pa
#     'P_inlet': 101325,     # Pa
#     'D_h': 0.01,           # m
#     'rho': 1000,           # kg/m³
#     'U': 1.0,              # m/s
#     'L': 1.0,              # m
#     'mu': 0.001,           # Pa.s
#     'Pr': 7.0,             # Prandtl number for water at room temp
#     'h': 500,              # W/m²K
#     'k': 0.6               # W/mK
# }
#
# col1, col2 = st.columns(2)
# with col1:
#     P_outlet = st.number_input("Outlet pressure (P_outlet) in Pascals", value=default_values['P_outlet'])
#     P_inlet = st.number_input("Inlet pressure (P_inlet) in Pascals", value=default_values['P_inlet'])
#     D_h = st.number_input("Hydraulic diameter (D_h) in meters", value=default_values['D_h'])
#     rho = st.number_input("Fluid density (rho) in kg/m³", value=default_values['rho'])
#     U = st.number_input("Mean velocity (U) in m/s", value=default_values['U'])
#
# with col2:
#     L = st.number_input("Channel length (L) in meters", value=default_values['L'])
#     mu = st.number_input("Dynamic viscosity (mu) in Pa.s", value=default_values['mu'])
#     Pr = st.number_input("Prandtl number (Pr)", value=default_values['Pr'])
#     h = st.number_input("Surface heat transfer coefficient (h) in W/m²K", value=default_values['h'])
#     k = st.number_input("Thermal conductivity (k) in W/mK", value=default_values['k'])
#
# # --- Calculate Button ---
# if st.button("Calculate and Log Data"):
#     # --- Calculations ---
#     try:
#         delta_p = P_outlet - P_inlet
#         Re = (rho * U * D_h) / mu
#         if Re <= 0:
#             st.error("Invalid Reynolds number. Please check your inputs.")
#             st.stop()
#
#         # Determine if flow is laminar or turbulent
#         if Re < 2300:
#             flow_type = 'Laminar'
#             f = 64 / Re  # Friction factor for laminar flow
#             Nu = 3.66    # Nusselt number for fully developed laminar flow in a circular pipe with constant wall temperature
#             f_o_smooth = f  # For comparison purposes, set f_o_smooth equal to f
#             Nu_o_smooth = Nu  # For comparison purposes, set Nu_o_smooth equal to Nu
#         else:
#             flow_type = 'Turbulent'
#             # Friction factor for turbulent flow (explicit formula)
#             f_o_smooth = 1 / ( (1.82 * math.log10(Re) - 1.64) ** 2 )
#             f = (2 * abs(delta_p) * D_h) / (rho * U**2 * L)
#             # Nusselt number for turbulent flow using the Gnielinski correlation
#             Pr_term = (Pr ** (1/3))
#             numerator = (f_o_smooth / 8) * (Re - 1000) * Pr
#             denominator = 1 + 12.7 * math.sqrt(f_o_smooth / 8) * (Pr_term - 1)
#             Nu_o_smooth = numerator / denominator
#             Nu = (h * D_h) / k  # Use hydraulic diameter for Nusselt number calculation
#
#         if f <= 0 or f_o_smooth <= 0 or Nu_o_smooth <= 0:
#             st.error("Friction factor or Nusselt number calculation resulted in non-positive value.")
#             st.stop()
#
#         TPF = (Nu / Nu_o_smooth) / ((f / f_o_smooth) ** (1/3))
#
#         # --- Append Data to Session State ---
#         data_row = {
#             'P_outlet': P_outlet,
#             'P_inlet': P_inlet,
#             'delta_p': delta_p,
#             'D_h': D_h,
#             'rho': rho,
#             'U': U,
#             'L': L,
#             'mu': mu,
#             'Pr': Pr,
#             'h': h,
#             'k': k,
#             'Re': Re,
#             'Flow Type': flow_type,
#             'f': f,
#             'f_o_smooth': f_o_smooth,
#             'Nu': Nu,
#             'Nu_o_smooth': Nu_o_smooth,
#             'TPF': TPF
#         }
#         # st.session_state['data'] = st.session_state['data'].append(data_row, ignore_index=True)
#         st.session_state['data'] = pd.concat([st.session_state['data'], pd.DataFrame([data_row])], ignore_index=True)
#
#         # --- Output ---
#         st.header("Calculated Parameters")
#         st.write(f"**Flow Type:** {flow_type}")
#         st.write(f"**Pressure drop (delta_p):** {delta_p:.2f} Pa")
#         st.write(f"**Reynolds number (Re):** {Re:.2f}")
#         st.write(f"**Friction factor (f):** {f:.5f}")
#         st.write(f"**Smooth friction factor (f_o,smooth):** {f_o_smooth:.5f}")
#         st.write(f"**Smooth Nusselt number (Nu_o,smooth):** {Nu_o_smooth:.2f}")
#         st.write(f"**Nusselt number (Nu):** {Nu:.2f}")
#         st.write(f"**TPF (Thermal Performance Factor):** {TPF:.4f}")
#
#     except ZeroDivisionError:
#         st.error("Division by zero occurred in calculations. Please check your inputs.")
#     except ValueError as e:
#         st.error(f"An error occurred during calculations: {e}")
#
# # --- Display Data Table ---
# st.header("Data Table")
# if not st.session_state['data'].empty:
#     st.write("All calculated data:")
#     st.dataframe(st.session_state['data'])
#
#     # --- Download Data ---
#     csv_data = st.session_state['data'].to_csv(index=False)
#     st.download_button(
#         label="Download Data as CSV",
#         data=csv_data,
#         file_name='calculated_data.csv',
#         mime='text/csv',
#     )
# else:
#     st.write("No data available yet. Perform calculations to generate data.")
#
# # --- Plotting Section ---
# st.header("Visualization")
# if not st.session_state['data'].empty:
#     st.write("""
#     Select the parameters you want to plot and choose the data points to include.
#     """)
#
#     # --- Select X and Y Axes ---
#     numeric_columns = st.session_state['data'].select_dtypes(include=np.number).columns.tolist()
#     x_axis = st.selectbox("Select X-axis parameter", options=numeric_columns)
#     y_axis = st.selectbox("Select Y-axis parameter", options=numeric_columns, index=1)
#
#     # --- Select Rows ---
#     st.write("Select data points to include in the plot:")
#     data_indices = st.session_state['data'].index.tolist()
#     selected_rows = st.multiselect(
#         "Select rows (by index)",
#         options=data_indices,
#         default=data_indices  # Default is all rows selected
#     )
#
#     if selected_rows:
#         # --- Plotting ---
#         plot_data = st.session_state['data'].loc[selected_rows]
#         fig, ax = plt.subplots()
#         ax.plot(plot_data[x_axis], plot_data[y_axis], marker='o', linestyle='-')
#         ax.set_xlabel(x_axis)
#         ax.set_ylabel(y_axis)
#         ax.set_title(f"{y_axis} vs {x_axis}")
#         ax.grid(True)
#         st.pyplot(fig)
#     else:
#         st.write("No data points selected for plotting.")
# else:
#     st.write("No data available for plotting. Perform calculations to generate data.")


# import streamlit as st
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # --- Title ---
# st.title("Fluid Flow and Heat Transfer Calculator with Data Logging and Comparison")

# # --- Introduction ---
# st.write("""
# This application calculates various fluid flow and heat transfer parameters such as pressure drop, Reynolds number, friction factor, Nusselt number, and TPF (Thermal Performance Factor).

# Each calculation can be logged into different datasets, which you can download as CSV files. You can compare up to two datasets by plotting their data on the same graph, with options to select the graph style.
# """)

# # --- Initialize Session State ---
# if 'datasets' not in st.session_state:
#     # Datasets will be a dictionary with dataset names as keys and DataFrames as values
#     st.session_state['datasets'] = {}

# # --- Input Section ---
# st.header("Input Parameters")

# # Define default values
# default_values = {
#     'P_outlet': 101325,    # Pa
#     'P_inlet': 101325,     # Pa
#     'D_h': 0.01,           # m
#     'rho': 1000,           # kg/m³
#     'U': 1.0,              # m/s
#     'L': 1.0,              # m
#     'mu': 0.001,           # Pa.s
#     'Pr': 7.0,             # Prandtl number for water at room temp
#     'h': 500,              # W/m²K
#     'k': 0.6               # W/mK
# }

# col1, col2 = st.columns(2)
# with col1:
#     P_outlet = st.number_input("Outlet pressure (P_outlet) in Pascals", value=default_values['P_outlet'])
#     P_inlet = st.number_input("Inlet pressure (P_inlet) in Pascals", value=default_values['P_inlet'])
#     D_h = st.number_input("Hydraulic diameter (D_h) in meters", value=default_values['D_h'])
#     rho = st.number_input("Fluid density (rho) in kg/m³", value=default_values['rho'])
#     U = st.number_input("Mean velocity (U) in m/s", value=default_values['U'])

# with col2:
#     L = st.number_input("Channel length (L) in meters", value=default_values['L'])
#     mu = st.number_input("Dynamic viscosity (mu) in Pa.s", value=default_values['mu'])
#     Pr = st.number_input("Prandtl number (Pr)", value=default_values['Pr'])
#     h = st.number_input("Surface heat transfer coefficient (h) in W/m²K", value=default_values['h'])
#     k = st.number_input("Thermal conductivity (k) in W/mK", value=default_values['k'])

# # --- Dataset Selection ---
# st.subheader("Select Dataset to Log Data")
# dataset_name = st.text_input("Enter dataset name (existing or new)", value="Dataset 1")

# # --- Calculate Button ---
# if st.button("Calculate and Log Data"):
#     # --- Calculations ---
#     try:
#         delta_p = P_outlet - P_inlet
#         Re = (rho * U * D_h) / mu
#         if Re <= 0:
#             st.error("Invalid Reynolds number. Please check your inputs.")
#             st.stop()

#         # Determine if flow is laminar or turbulent
#         if Re < 2300:
#             flow_type = 'Laminar'
#             f = 64 / Re  # Friction factor for laminar flow
#             Nu = 3.66    # Nusselt number for fully developed laminar flow in a circular pipe with constant wall temperature
#             f_o_smooth = f  # For comparison purposes, set f_o_smooth equal to f
#             Nu_o_smooth = Nu  # For comparison purposes, set Nu_o_smooth equal to Nu
#         else:
#             flow_type = 'Turbulent'
#             # Friction factor for turbulent flow (explicit formula)
#             f_o_smooth = 1 / ( (1.82 * math.log10(Re) - 1.64) ** 2 )
#             f = (2 * abs(delta_p) * D_h) / (rho * U**2 * L)
#             # Nusselt number for turbulent flow using the Gnielinski correlation
#             Pr_term = (Pr ** (1/3))
#             numerator = (f_o_smooth / 8) * (Re - 1000) * Pr
#             denominator = 1 + 12.7 * math.sqrt(f_o_smooth / 8) * (Pr_term - 1)
#             Nu_o_smooth = numerator / denominator
#             Nu = (h * D_h) / k  # Use hydraulic diameter for Nusselt number calculation

#         if f <= 0 or f_o_smooth <= 0 or Nu_o_smooth <= 0:
#             st.error("Friction factor or Nusselt number calculation resulted in non-positive value.")
#             st.stop()

#         TPF = (Nu / Nu_o_smooth) / ((f / f_o_smooth) ** (1/3))

#         # --- Append Data to Dataset ---
#         data_row = {
#             'P_outlet': P_outlet,
#             'P_inlet': P_inlet,
#             'delta_p': delta_p,
#             'D_h': D_h,
#             'rho': rho,
#             'U': U,
#             'L': L,
#             'mu': mu,
#             'Pr': Pr,
#             'h': h,
#             'k': k,
#             'Re': Re,
#             'Flow Type': flow_type,
#             'f': f,
#             'f_o_smooth': f_o_smooth,
#             'Nu': Nu,
#             'Nu_o_smooth': Nu_o_smooth,
#             'TPF': TPF
#         }

#         if dataset_name in st.session_state['datasets']:
#             st.session_state['datasets'][dataset_name] = pd.concat(
#                 [st.session_state['datasets'][dataset_name], pd.DataFrame([data_row])],
#                 ignore_index=True
#             )
#         else:
#             st.session_state['datasets'][dataset_name] = pd.DataFrame([data_row])

#         # --- Output ---
#         st.header("Calculated Parameters")
#         st.write(f"**Flow Type:** {flow_type}")
#         st.write(f"**Pressure drop (delta_p):** {delta_p:.2f} Pa")
#         st.write(f"**Reynolds number (Re):** {Re:.2f}")
#         st.write(f"**Friction factor (f):** {f:.5f}")
#         st.write(f"**Smooth friction factor (f_o,smooth):** {f_o_smooth:.5f}")
#         st.write(f"**Smooth Nusselt number (Nu_o,smooth):** {Nu_o_smooth:.2f}")
#         st.write(f"**Nusselt number (Nu):** {Nu:.2f}")
#         st.write(f"**TPF (Thermal Performance Factor):** {TPF:.4f}")

#         st.success(f"Data logged to {dataset_name}.")

#     except ZeroDivisionError:
#         st.error("Division by zero occurred in calculations. Please check your inputs.")
#     except ValueError as e:
#         st.error(f"An error occurred during calculations: {e}")

# # --- Display Data Tables ---
# st.header("Data Tables")
# if st.session_state['datasets']:
#     for name, data in st.session_state['datasets'].items():
#         st.subheader(f"Dataset: {name}")
#         st.dataframe(data)

#         # --- Download Data ---
#         csv_data = data.to_csv(index=False)
#         st.download_button(
#             label=f"Download {name} as CSV",
#             data=csv_data,
#             file_name=f'{name}.csv',
#             mime='text/csv',
#             key=f'download-{name}'
#         )
# else:
#     st.write("No datasets available yet. Perform calculations to generate data.")

# # --- Plotting Section ---
# st.header("Visualization")
# if st.session_state['datasets']:
#     st.write("""
#     Select up to two datasets and parameters you want to plot. You can compare them on the same graph with different colors.
#     """)
#     dataset_names = list(st.session_state['datasets'].keys())

#     # --- Select Datasets ---
#     selected_datasets = st.multiselect("Select datasets to include in the plot (max 2)", options=dataset_names, default=dataset_names[:2], key='dataset-select', max_selections=2)
#     if selected_datasets:
#         # --- Select X and Y Axes ---
#         # Get common numeric columns
#         numeric_columns = None
#         for ds in selected_datasets:
#             cols = st.session_state['datasets'][ds].select_dtypes(include=np.number).columns.tolist()
#             if numeric_columns is None:
#                 numeric_columns = set(cols)
#             else:
#                 numeric_columns = numeric_columns.intersection(cols)
#         numeric_columns = list(numeric_columns)
#         if not numeric_columns:
#             st.error("No common numeric columns available for plotting.")
#         else:
#             x_axis = st.selectbox("Select X-axis parameter", options=numeric_columns)
#             y_axis = st.selectbox("Select Y-axis parameter", options=numeric_columns, index=1)

#             # --- Select Graph Style ---
#             graph_styles = ['Line', 'Scatter', 'Bar']
#             graph_style = st.selectbox("Select graph style", options=graph_styles)

#             # --- Plotting ---
#             fig, ax = plt.subplots()

#             colors = ['blue', 'red', 'green', 'purple', 'orange']
#             for idx, ds_name in enumerate(selected_datasets):
#                 data = st.session_state['datasets'][ds_name]
#                 data_indices = data.index.tolist()
#                 selected_rows = st.multiselect(
#                     f"Select rows from {ds_name} (by index)",
#                     options=data_indices,
#                     default=data_indices,
#                     key=f'select-{ds_name}'
#                 )
#                 plot_data = data.loc[selected_rows]

#                 if graph_style == 'Line':
#                     ax.plot(plot_data[x_axis], plot_data[y_axis], marker='o', linestyle='-', color=colors[idx % len(colors)], label=ds_name)
#                 elif graph_style == 'Scatter':
#                     ax.scatter(plot_data[x_axis], plot_data[y_axis], color=colors[idx % len(colors)], label=ds_name)
#                 elif graph_style == 'Bar':
#                     ax.bar(plot_data[x_axis] + idx*0.1, plot_data[y_axis], width=0.1, color=colors[idx % len(colors)], label=ds_name)

#             ax.set_xlabel(x_axis)
#             ax.set_ylabel(y_axis)
#             ax.set_title(f"{y_axis} vs {x_axis}")
#             ax.grid(True)
#             ax.legend()
#             st.pyplot(fig)
#     else:
#         st.write("Select at least one dataset to plot.")
# else:
#     st.write("No data available for plotting. Perform calculations to generate data.")





# import streamlit as st
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # --- Title ---
# st.title("Fluid Flow and Heat Transfer Calculator with Data Logging and Comparison")

# # --- Introduction ---


# # --- Initialize Session State ---
# if 'datasets' not in st.session_state:
#     st.session_state['datasets'] = {}

# # --- Input Section ---
# st.header("Input Parameters")

# # Define default values as floats
# default_values = {
#     'P_outlet': 500.0,    # Pa
#     'P_inlet': 1000.0,     # Pa
#     'D_h': 0.04,           # m, hydraulic diameter of the pipe
#     'rho': 1.0,            # kg/m³, density of air at room temperature
#     'U': 1.0,              # m/s, mean velocity (assumed)
#     'L': 0.42,              # m, channel length (400 mm converted to m)
#     'mu': 3.178e-5,         # Pa.s, dynamic viscosity of air
#     'Pr': 0.71,            # dimensionless, Prandtl number for air
#     'h': 500.0,            # W/m²K, surface heat transfer coefficient (assumed)
#     'k': 0.045             # W/mK, thermal conductivity of air
# }

# # Define a very large range as floats
# LARGE_MIN = -1e12
# LARGE_MAX = 1e12

# col1, col2 = st.columns(2)
# with col1:
#     P_outlet = st.number_input(
#         "Outlet pressure (P_outlet) in Pascals",
#         value=default_values['P_outlet'],
#         min_value=float(LARGE_MIN), max_value=float(LARGE_MAX), step=1.0
#     )
#     P_inlet = st.number_input(
#         "Inlet pressure (P_inlet) in Pascals",
#         value=default_values['P_inlet'],
#         min_value=float(LARGE_MIN), max_value=float(LARGE_MAX), step=1.0
#     )
#     D_h = st.number_input(
#         "Hydraulic diameter (D_h) in meters",
#         value=default_values['D_h'],
#         min_value=0.0, max_value=float(LARGE_MAX), step=0.001
#     )
#     rho = st.number_input(
#         "Fluid density (rho) in kg/m³",
#         value=default_values['rho'],
#         min_value=0.0, max_value=float(LARGE_MAX), step=1.0
#     )
#     U = st.number_input(
#         "Mean velocity (U) in m/s",
#         value=default_values['U'],
#         min_value=float(LARGE_MIN), max_value=float(LARGE_MAX), step=0.1
#     )

# with col2:
#     L = st.number_input(
#         "Channel length (L) in meters",
#         value=default_values['L'],
#         min_value=0.0, max_value=float(LARGE_MAX), step=0.1
#     )
#     mu = st.number_input(
#         "Dynamic viscosity (mu) in Pa.s",
#         value=default_values['mu'],
#         min_value=0.0, max_value=float(LARGE_MAX), step=1e-6, format="%.6e"
#     )
#     Pr = st.number_input(
#         "Prandtl number (Pr)",
#         value=default_values['Pr'],
#         min_value=0.0, max_value=float(LARGE_MAX), step=0.01
#     )
#     h = st.number_input(
#         "Surface heat transfer coefficient (h) in W/m²K",
#         value=default_values['h'],
#         min_value=0.0, max_value=float(LARGE_MAX), step=1.0
#     )
#     k = st.number_input(
#         "Thermal conductivity (k) in W/mK",
#         value=default_values['k'],
#         min_value=0.0, max_value=float(LARGE_MAX), step=0.001
#     )

# # --- Dataset Selection ---
# st.subheader("Select Dataset to Log Data")
# dataset_name = st.text_input("Enter dataset name (existing or new)", value="Dataset 1")

# # --- Calculate Button ---
# if st.button("Calculate and Log Data"):
#     try:
#         delta_p = P_outlet - P_inlet
#         Re = (rho * U * D_h) / mu
#         if Re <= 0:
#             st.error("Invalid Reynolds number. Please check your inputs.")
#             st.stop()

#         flow_type = 'Turbulent'
#         # Friction factor for turbulent flow (explicit formula for smooth pipes)
#         f_o_smooth = 1 / ((1.82 * math.log10(Re) - 1.64) ** 2)
#         f = (2 * abs(delta_p) * D_h) / (rho * U**2 * L)
#         # Nusselt number for turbulent flow using the Gnielinski correlation
#         Pr_term = Pr ** (1/3)
#         numerator = (f_o_smooth / 8) * (Re - 1000) * Pr
#         denominator = 1 + 12.7 * math.sqrt(f_o_smooth / 8) * (Pr_term - 1)
#         Nu_o_smooth = numerator / denominator
#         Nu = (h * D_h) / k  # Calculated based on the given h and k

#         if f <= 0 or f_o_smooth <= 0 or Nu_o_smooth <= 0:
#             st.error("Friction factor or Nusselt number calculation resulted in non-positive value.")
#             st.stop()

#         TPF = (Nu / Nu_o_smooth) / ((f / f_o_smooth) ** (1/3))

#         # --- Append Data to Dataset ---
#         data_row = {
#             'P_outlet': P_outlet,
#             'P_inlet': P_inlet,
#             'delta_p': delta_p,
#             'D_h': D_h,
#             'rho': rho,
#             'U': U,
#             'L': L,
#             'mu': mu,
#             'Pr': Pr,
#             'h': h,
#             'k': k,
#             'Re': Re,
#             'Flow Type': flow_type,
#             'f': f,
#             'f_o_smooth': f_o_smooth,
#             'Nu': Nu,
#             'Nu_o_smooth': Nu_o_smooth,
#             'TPF': TPF
#         }

#         if dataset_name in st.session_state['datasets']:
#             st.session_state['datasets'][dataset_name] = pd.concat(
#                 [st.session_state['datasets'][dataset_name], pd.DataFrame([data_row])],
#                 ignore_index=True
#             )
#         else:
#             st.session_state['datasets'][dataset_name] = pd.DataFrame([data_row])

#         st.header("Calculated Parameters")
#         st.write(f"**Flow Type:** {flow_type}")
#         st.write(f"**Pressure drop (delta_p):** {delta_p:.2f} Pa")
#         st.write(f"**Reynolds number (Re):** {Re:.2f}")
#         st.write(f"**Friction factor (f):** {f:.5f}")
#         st.write(f"**Smooth friction factor (f_o,smooth):** {f_o_smooth:.5f}")
#         st.write(f"**Smooth Nusselt number (Nu_o,smooth):** {Nu_o_smooth:.2f}")
#         st.write(f"**Nusselt number (Nu):** {Nu:.2f}")
#         st.write(f"**TPF (Thermal Performance Factor):** {TPF:.4f}")

#         st.success(f"Data logged to {dataset_name}.")

#     except ZeroDivisionError:
#         st.error("Division by zero occurred in calculations. Please check your inputs.")
#     except ValueError as e:
#         st.error(f"An error occurred during calculations: {e}")

# # --- Display Data Tables ---
# st.header("Data Tables")
# if st.session_state['datasets']:
#     for name, data in st.session_state['datasets'].items():
#         st.subheader(f"Dataset: {name}")
#         st.dataframe(data)

#         csv_data = data.to_csv(index=False)
#         st.download_button(
#             label=f"Download {name} as CSV",
#             data=csv_data,
#             file_name=f'{name}.csv',
#             mime='text/csv',
#             key=f'download-{name}'
#         )
# else:
#     st.write("No datasets available yet. Perform calculations to generate data.")

# # --- Plotting Section ---
# st.header("Visualization")
# if st.session_state['datasets']:
#     st.write("""
#     Select up to two datasets and parameters you want to plot. You can compare them on the same graph with different colors.
#     """)
#     dataset_names = list(st.session_state['datasets'].keys())
#     selected_datasets = st.multiselect("Select datasets to include in the plot (max 2)", options=dataset_names, default=dataset_names[:2], key='dataset-select', max_selections=2)
#     if selected_datasets:
#         numeric_columns = None
#         for ds in selected_datasets:
#             cols = st.session_state['datasets'][ds].select_dtypes(include=np.number).columns.tolist()
#             if numeric_columns is None:
#                 numeric_columns = set(cols)
#             else:
#                 numeric_columns = numeric_columns.intersection(cols)
#         numeric_columns = list(numeric_columns)
#         if not numeric_columns:
#             st.error("No common numeric columns available for plotting.")
#         else:
#             x_axis = st.selectbox("Select X-axis parameter", options=numeric_columns)
#             y_axis = st.selectbox("Select Y-axis parameter", options=numeric_columns, index=1)
#             graph_styles = ['Line', 'Scatter', 'Bar']
#             graph_style = st.selectbox("Select graph style", options=graph_styles)

#             fig, ax = plt.subplots()
#             colors = ['blue', 'red', 'green', 'purple', 'orange']
#             for idx, ds_name in enumerate(selected_datasets):
#                 data = st.session_state['datasets'][ds_name]
#                 data_indices = data.index.tolist()
#                 selected_rows = st.multiselect(
#                     f"Select rows from {ds_name} (by index)",
#                     options=data_indices,
#                     default=data_indices,
#                     key=f'select-{ds_name}'
#                 )
#                 plot_data = data.loc[selected_rows]

#                 if graph_style == 'Line':
#                     ax.plot(plot_data[x_axis], plot_data[y_axis], marker='o', linestyle='-', color=colors[idx % len(colors)], label=ds_name)
#                 elif graph_style == 'Scatter':
#                     ax.scatter(plot_data[x_axis], plot_data[y_axis], color=colors[idx % len(colors)], label=ds_name)
#                 elif graph_style == 'Bar':
#                     ax.bar(plot_data[x_axis] + idx*0.1, plot_data[y_axis], width=0.1, color=colors[idx % len(colors)], label=ds_name)

#             ax.set_xlabel(x_axis)
#             ax.set_ylabel(y_axis)
#             ax.set_title(f"{y_axis} vs {x_axis}")
#             ax.grid(True)
#             ax.legend()
#             st.pyplot(fig)
#     else:
#         st.write("Select at least one dataset to plot.")
# else:
#     st.write("No data available for plotting. Perform calculations to generate data.")



# import streamlit as st
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # --- Title ---
# st.title("Fluid Flow and Heat Transfer Calculator with Data Logging and Comparison")

# # --- Initialize Session State ---
# if 'datasets' not in st.session_state:
#     st.session_state['datasets'] = {}

# # --- Input Section ---
# st.header("Input Parameters")

# # Define default values as floats
# default_values = {
#     'P_outlet': 500.0,    # Pa
#     'P_inlet': 1000.0,    # Pa
#     'D_h': 0.04,         # m, hydraulic diameter of the pipe
#     'rho': 1.2,          # kg/m³, density of air at room temperature
#     'U': 1.0,            # m/s, mean velocity (assumed)
#     'L': 0.4,            # m, channel length (400 mm converted to m)
#     'mu': 1.81e-5,       # Pa.s, dynamic viscosity of air
#     'Pr': 0.71,          # dimensionless, Prandtl number for air
#     'h': 500.0,          # W/m²K, surface heat transfer coefficient (assumed)
#     'k': 0.026           # W/mK, thermal conductivity of air
# }

# # Define a very large range as floats
# LARGE_MIN = -1e12
# LARGE_MAX = 1e12

# # Use a format that displays up to 10 decimals for higher precision
# PRECISION_FORMAT = "%.3f"

# col1, col2 = st.columns(2)
# with col1:
#     P_outlet = st.number_input(
#         "Outlet pressure (P_outlet) in Pascals",
#         value=default_values['P_outlet'],
#         min_value=float(LARGE_MIN),
#         max_value=float(LARGE_MAX),
#         step=1.0,
#         format=PRECISION_FORMAT
#     )
#     P_inlet = st.number_input(
#         "Inlet pressure (P_inlet) in Pascals",
#         value=default_values['P_inlet'],
#         min_value=float(LARGE_MIN),
#         max_value=float(LARGE_MAX),
#         step=1.0,
#         format=PRECISION_FORMAT
#     )
#     D_h = st.number_input(
#         "Hydraulic diameter (D_h) in meters",
#         value=default_values['D_h'],
#         min_value=0.0,
#         max_value=float(LARGE_MAX),
#         step=0.001,
#         format=PRECISION_FORMAT
#     )
#     rho = st.number_input(
#         "Fluid density (rho) in kg/m³",
#         value=default_values['rho'],
#         min_value=0.0,
#         max_value=float(LARGE_MAX),
#         step=1.0,
#         format=PRECISION_FORMAT
#     )
#     U = st.number_input(
#         "Mean velocity (U) in m/s",
#         value=default_values['U'],
#         min_value=float(LARGE_MIN),
#         max_value=float(LARGE_MAX),
#         step=0.1,
#         format=PRECISION_FORMAT
#     )

# with col2:
#     L = st.number_input(
#         "Channel length (L) in meters",
#         value=default_values['L'],
#         min_value=0.0,
#         max_value=float(LARGE_MAX),
#         step=0.1,
#         format=PRECISION_FORMAT
#     )
#     mu = st.number_input(
#         "Dynamic viscosity (mu) in Pa.s",
#         value=default_values['mu'],
#         min_value=0.0,
#         max_value=float(LARGE_MAX),
#         step=1e-6,
#         format="%.10e"  # Using scientific notation for very small numbers
#     )
#     Pr = st.number_input(
#         "Prandtl number (Pr)",
#         value=default_values['Pr'],
#         min_value=0.0,
#         max_value=float(LARGE_MAX),
#         step=0.01,
#         format=PRECISION_FORMAT
#     )
#     h = st.number_input(
#         "Surface heat transfer coefficient (h) in W/m²K",
#         value=default_values['h'],
#         min_value=0.0,
#         max_value=float(LARGE_MAX),
#         step=1.0,
#         format=PRECISION_FORMAT
#     )
#     k = st.number_input(
#         "Thermal conductivity (k) in W/mK",
#         value=default_values['k'],
#         min_value=0.0,
#         max_value=float(LARGE_MAX),
#         step=0.001,
#         format=PRECISION_FORMAT
#     )

# # --- Dataset Selection ---
# st.subheader("Select Dataset to Log Data")
# dataset_name = st.text_input("Enter dataset name (existing or new)", value="Dataset 1")

# # --- Calculate Button ---
# if st.button("Calculate and Log Data"):
#     try:
#         delta_p = P_outlet - P_inlet
#         Re = (rho * U * D_h) / mu
#         if Re <= 0:
#             st.error("Invalid Reynolds number. Please check your inputs.")
#             st.stop()

#         flow_type = 'Turbulent'
#         # Friction factor for turbulent flow (explicit formula for smooth pipes)
#         f_o_smooth = 1 / ((1.82 * math.log10(Re) - 1.64) ** 2)
#         f = (2 * abs(delta_p) * D_h) / (rho * U**2 * L)
#         # Nusselt number for turbulent flow using the Gnielinski correlation
#         Pr_term = Pr ** (1/3)
#         numerator = (f_o_smooth / 8) * (Re - 1000) * Pr
#         denominator = 1 + 12.7 * math.sqrt(f_o_smooth / 8) * (Pr_term - 1)
#         Nu_o_smooth = numerator / denominator
#         Nu = (h * D_h) / k  # Calculated based on the given h and k

#         if f <= 0 or f_o_smooth <= 0 or Nu_o_smooth <= 0:
#             st.error("Friction factor or Nusselt number calculation resulted in non-positive value.")
#             st.stop()

#         TPF = (Nu / Nu_o_smooth) / ((f / f_o_smooth) ** (1/3))

#         # --- Append Data to Dataset ---
#         data_row = {
#             'P_outlet': P_outlet,
#             'P_inlet': P_inlet,
#             'delta_p': delta_p,
#             'D_h': D_h,
#             'rho': rho,
#             'U': U,
#             'L': L,
#             'mu': mu,
#             'Pr': Pr,
#             'h': h,
#             'k': k,
#             'Re': Re,
#             'Flow Type': flow_type,
#             'f': f,
#             'f_o_smooth': f_o_smooth,
#             'Nu': Nu,
#             'Nu_o_smooth': Nu_o_smooth,
#             'TPF': TPF
#         }

#         if dataset_name in st.session_state['datasets']:
#             st.session_state['datasets'][dataset_name] = pd.concat(
#                 [st.session_state['datasets'][dataset_name], pd.DataFrame([data_row])],
#                 ignore_index=True
#             )
#         else:
#             st.session_state['datasets'][dataset_name] = pd.DataFrame([data_row])

#         st.header("Calculated Parameters")
#         st.write(f"**Flow Type:** {flow_type}")
#         st.write(f"**Pressure drop (delta_p):** {delta_p:.10f} Pa")
#         st.write(f"**Reynolds number (Re):** {Re:.10f}")
#         st.write(f"**Friction factor (f):** {f:.10f}")
#         st.write(f"**Smooth friction factor (f_o,smooth):** {f_o_smooth:.10f}")
#         st.write(f"**Smooth Nusselt number (Nu_o,smooth):** {Nu_o_smooth:.10f}")
#         st.write(f"**Nusselt number (Nu):** {Nu:.10f}")
#         st.write(f"**TPF (Thermal Performance Factor):** {TPF:.10f}")

#         st.success(f"Data logged to {dataset_name}.")

#     except ZeroDivisionError:
#         st.error("Division by zero occurred in calculations. Please check your inputs.")
#     except ValueError as e:
#         st.error(f"An error occurred during calculations: {e}")

# # --- Display Data Tables ---
# st.header("Data Tables")
# if st.session_state['datasets']:
#     for name, data in st.session_state['datasets'].items():
#         st.subheader(f"Dataset: {name}")
#         st.dataframe(data)

#         csv_data = data.to_csv(index=False)
#         st.download_button(
#             label=f"Download {name} as CSV",
#             data=csv_data,
#             file_name=f'{name}.csv',
#             mime='text/csv',
#             key=f'download-{name}'
#         )
# else:
#     st.write("No datasets available yet. Perform calculations to generate data.")

# # --- Plotting Section ---
# st.header("Visualization")
# if st.session_state['datasets']:
#     st.write("""
#     Select up to two datasets and parameters you want to plot. You can compare them on the same graph with different colors.
#     """)
#     dataset_names = list(st.session_state['datasets'].keys())
#     selected_datasets = st.multiselect("Select datasets to include in the plot (max 2)", options=dataset_names, default=dataset_names[:2], key='dataset-select', max_selections=2)
#     if selected_datasets:
#         numeric_columns = None
#         for ds in selected_datasets:
#             cols = st.session_state['datasets'][ds].select_dtypes(include=np.number).columns.tolist()
#             if numeric_columns is None:
#                 numeric_columns = set(cols)
#             else:
#                 numeric_columns = numeric_columns.intersection(cols)
#         numeric_columns = list(numeric_columns)
#         if not numeric_columns:
#             st.error("No common numeric columns available for plotting.")
#         else:
#             x_axis = st.selectbox("Select X-axis parameter", options=numeric_columns)
#             y_axis = st.selectbox("Select Y-axis parameter", options=numeric_columns, index=1)
#             graph_styles = ['Line', 'Scatter', 'Bar']
#             graph_style = st.selectbox("Select graph style", options=graph_styles)

#             fig, ax = plt.subplots()
#             colors = ['blue', 'red', 'green', 'purple', 'orange']
#             for idx, ds_name in enumerate(selected_datasets):
#                 data = st.session_state['datasets'][ds_name]
#                 data_indices = data.index.tolist()
#                 selected_rows = st.multiselect(
#                     f"Select rows from {ds_name} (by index)",
#                     options=data_indices,
#                     default=data_indices,
#                     key=f'select-{ds_name}'
#                 )
#                 plot_data = data.loc[selected_rows]

#                 if graph_style == 'Line':
#                     ax.plot(plot_data[x_axis], plot_data[y_axis], marker='o', linestyle='-', color=colors[idx % len(colors)], label=ds_name)
#                 elif graph_style == 'Scatter':
#                     ax.scatter(plot_data[x_axis], plot_data[y_axis], color=colors[idx % len(colors)], label=ds_name)
#                 elif graph_style == 'Bar':
#                     ax.bar(plot_data[x_axis] + idx*0.1, plot_data[y_axis], width=0.1, color=colors[idx % len(colors)], label=ds_name)

#             ax.set_xlabel(x_axis)
#             ax.set_ylabel(y_axis)
#             ax.set_title(f"{y_axis} vs {x_axis}")
#             ax.grid(True)
#             ax.legend()
#             st.pyplot(fig)
#     else:
#         st.write("Select at least one dataset to plot.")
# else:
#     st.write("No data available for plotting. Perform calculations to generate data.")


import streamlit as st
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Title ---
st.title("Fluid Flow and Heat Transfer Calculator with Data Logging and Comparison")

# --- Initialize Session State ---
if 'datasets' not in st.session_state:
    st.session_state['datasets'] = {}

# --- Input Section ---
st.header("Input Parameters")

# Define default values as floats
default_values = {
    'P_outlet': 500.0,    # Pa
    'P_inlet': 1000.0,    # Pa
    'D_h': 0.04,         # m, hydraulic diameter of the pipe
    'rho': 1.2,          # kg/m³, density of air at room temperature
    'U': 1.0,            # m/s, mean velocity (assumed)
    'L': 0.4,            # m, channel length (400 mm converted to m)
    'mu': 1.81e-5,       # Pa.s, dynamic viscosity of air
    'Pr': 0.71,          # dimensionless, Prandtl number for air
    'h': 500.0,          # W/m²K, surface heat transfer coefficient (assumed)
    'k': 0.026           # W/mK, thermal conductivity of air
}

# Define a very large range as floats
LARGE_MIN = -1e12
LARGE_MAX = 1e12

# You can choose to display more or fewer decimals for the input fields.
# Here we use a reasonable display format.
col1, col2 = st.columns(2)
with col1:
    P_outlet = st.number_input(
        "Outlet pressure (P_outlet) in Pascals",
        value=default_values['P_outlet'],
        min_value=float(LARGE_MIN), max_value=float(LARGE_MAX),
        step=1.0,
        format="%.2f"
    )
    P_inlet = st.number_input(
        "Inlet pressure (P_inlet) in Pascals",
        value=default_values['P_inlet'],
        min_value=float(LARGE_MIN), max_value=float(LARGE_MAX),
        step=1.0,
        format="%.2f"
    )
    D_h = st.number_input(
        "Hydraulic diameter (D_h) in meters",
        value=default_values['D_h'],
        min_value=0.0, max_value=float(LARGE_MAX),
        step=0.001,
        format="%.3f"
    )
    rho = st.number_input(
        "Fluid density (rho) in kg/m³",
        value=default_values['rho'],
        min_value=0.0, max_value=float(LARGE_MAX),
        step=0.1,
        format="%.2f"
    )
    U = st.number_input(
        "Mean velocity (U) in m/s",
        value=default_values['U'],
        min_value=float(LARGE_MIN), max_value=float(LARGE_MAX),
        step=0.1,
        format="%.2f"
    )

with col2:
    L = st.number_input(
        "Channel length (L) in meters",
        value=default_values['L'],
        min_value=0.0, max_value=float(LARGE_MAX),
        step=0.1,
        format="%.2f"
    )
    mu = st.number_input(
        "Dynamic viscosity (mu) in Pa.s",
        value=default_values['mu'],
        min_value=0.0, max_value=float(LARGE_MAX),
        step=1e-6,
        format="%.6e"  # Scientific notation for small numbers
    )
    Pr = st.number_input(
        "Prandtl number (Pr)",
        value=default_values['Pr'],
        min_value=0.0, max_value=float(LARGE_MAX),
        step=0.01,
        format="%.2f"
    )
    h = st.number_input(
        "Surface heat transfer coefficient (h) in W/m²K",
        value=default_values['h'],
        min_value=0.0, max_value=float(LARGE_MAX),
        step=1.0,
        format="%.2f"
    )
    k = st.number_input(
        "Thermal conductivity (k) in W/mK",
        value=default_values['k'],
        min_value=0.0, max_value=float(LARGE_MAX),
        step=0.001,
        format="%.3f"
    )

# --- Dataset Selection ---
st.subheader("Select Dataset to Log Data")
dataset_name = st.text_input("Enter dataset name (existing or new)", value="Dataset 1")

# --- Calculate Button ---
if st.button("Calculate and Log Data"):
    try:
        # Calculate parameters
        delta_p = P_outlet - P_inlet
        Re = (rho * U * D_h) / mu
        if Re <= 0:
            st.error("Invalid Reynolds number. Please check your inputs.")
            st.stop()

        flow_type = 'Turbulent'
        # Friction factor for turbulent flow (explicit formula for smooth pipes)
        f_o_smooth = 1 / ((1.82 * math.log10(Re) - 1.64) ** 2)
        f = (2 * abs(delta_p) * D_h) / (rho * U**2 * L)
        # Nusselt number for turbulent flow using the Gnielinski correlation
        Pr_term = Pr ** (1/3)
        numerator = (f_o_smooth / 8) * (Re - 1000) * Pr
        denominator = 1 + 12.7 * math.sqrt(f_o_smooth / 8) * (Pr_term - 1)
        Nu_o_smooth = numerator / denominator
        Nu = (h * D_h) / k  # Based on given h and k

        if f <= 0 or f_o_smooth <= 0 or Nu_o_smooth <= 0:
            st.error("Friction factor or Nusselt number calculation resulted in non-positive value.")
            st.stop()

        TPF = (Nu / Nu_o_smooth) / ((f / f_o_smooth) ** (1/3))

        # --- Append Data to Dataset ---
        # The logged data is rounded as specified before logging.
        data_row = {
            'P_outlet (Pa)': round(P_outlet, 2),
            'P_inlet (Pa)': round(P_inlet, 2),
            'delta_p (Pa)': round(delta_p, 2),
            'D_h (m)': round(D_h, 3),
            'rho (kg/m³)': round(rho, 2),
            'U (m/s)': round(U, 2),
            'L (m)': round(L, 2),
            'mu (Pa.s)': mu,  # For small values, using scientific notation is clearer.
            'Pr': round(Pr, 2),
            'h (W/m²K)': round(h, 2),
            'k (W/mK)': round(k, 3),
            'Re': round(Re, 2),
            'Flow Type': flow_type,
            'f': round(f, 5),
            'f_o_smooth': round(f_o_smooth, 5),
            'Nu': round(Nu, 3),
            'Nu_o_smooth': round(Nu_o_smooth, 3),
            'TPF': round(TPF, 4)
        }

        if dataset_name in st.session_state['datasets']:
            st.session_state['datasets'][dataset_name] = pd.concat(
                [st.session_state['datasets'][dataset_name], pd.DataFrame([data_row])],
                ignore_index=True
            )
        else:
            st.session_state['datasets'][dataset_name] = pd.DataFrame([data_row])

        # --- Display Calculated Parameters ---
        st.header("Calculated Parameters")
        st.write(f"**Flow Type:** {flow_type}")
        st.write(f"**Pressure drop (delta_p):** {delta_p:.2f} Pa")
        st.write(f"**Reynolds number (Re):** {Re:.2f}")
        st.write(f"**Friction factor (f):** {f:.5f}")
        st.write(f"**Smooth friction factor (f_o,smooth):** {f_o_smooth:.5f}")
        st.write(f"**Smooth Nusselt number (Nu_o,smooth):** {Nu_o_smooth:.3f}")
        st.write(f"**Nusselt number (Nu):** {Nu:.3f}")
        st.write(f"**TPF (Thermal Performance Factor):** {TPF:.4f}")

        st.success(f"Data logged to {dataset_name}.")

    except ZeroDivisionError:
        st.error("Division by zero occurred in calculations. Please check your inputs.")
    except ValueError as e:
        st.error(f"An error occurred during calculations: {e}")

# --- Display Data Tables ---
st.header("Data Tables")
if st.session_state['datasets']:
    for name, data in st.session_state['datasets'].items():
        st.subheader(f"Dataset: {name}")
        st.dataframe(data)

        csv_data = data.to_csv(index=False)
        st.download_button(
            label=f"Download {name} as CSV",
            data=csv_data,
            file_name=f'{name}.csv',
            mime='text/csv',
            key=f'download-{name}'
        )
else:
    st.write("No datasets available yet. Perform calculations to generate data.")

# --- Plotting Section ---
st.header("Visualization")
if st.session_state['datasets']:
    st.write("""
    Select up to two datasets and parameters you want to plot. You can compare them on the same graph with different colors.
    """)
    dataset_names = list(st.session_state['datasets'].keys())
    selected_datasets = st.multiselect("Select datasets to include in the plot (max 2)", options=dataset_names, default=dataset_names[:2], key='dataset-select', max_selections=2)
    if selected_datasets:
        numeric_columns = None
        for ds in selected_datasets:
            cols = st.session_state['datasets'][ds].select_dtypes(include=np.number).columns.tolist()
            if numeric_columns is None:
                numeric_columns = set(cols)
            else:
                numeric_columns = numeric_columns.intersection(cols)
        numeric_columns = list(numeric_columns)
        if not numeric_columns:
            st.error("No common numeric columns available for plotting.")
        else:
            x_axis = st.selectbox("Select X-axis parameter", options=numeric_columns)
            y_axis = st.selectbox("Select Y-axis parameter", options=numeric_columns, index=1)
            graph_styles = ['Line', 'Scatter', 'Bar']
            graph_style = st.selectbox("Select graph style", options=graph_styles)

            fig, ax = plt.subplots()
            colors = ['blue', 'red', 'green', 'purple', 'orange']
            for idx, ds_name in enumerate(selected_datasets):
                data = st.session_state['datasets'][ds_name]
                data_indices = data.index.tolist()
                selected_rows = st.multiselect(
                    f"Select rows from {ds_name} (by index)",
                    options=data_indices,
                    default=data_indices,
                    key=f'select-{ds_name}'
                )
                plot_data = data.loc[selected_rows]

                if graph_style == 'Line':
                    ax.plot(plot_data[x_axis], plot_data[y_axis], marker='o', linestyle='-', color=colors[idx % len(colors)], label=ds_name)
                elif graph_style == 'Scatter':
                    ax.scatter(plot_data[x_axis], plot_data[y_axis], color=colors[idx % len(colors)], label=ds_name)
                elif graph_style == 'Bar':
                    ax.bar(plot_data[x_axis] + idx*0.1, plot_data[y_axis], width=0.1, color=colors[idx % len(colors)], label=ds_name)

            ax.set_xlabel(x_axis)
            ax.set_ylabel(y_axis)
            ax.set_title(f"{y_axis} vs {x_axis}")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
    else:
        st.write("Select at least one dataset to plot.")
else:
    st.write("No data available for plotting. Perform calculations to generate data.")
