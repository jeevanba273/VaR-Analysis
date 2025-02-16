# Value-at-Risk (VaR) Analysis App

Welcome to the **Value-at-Risk (VaR) Analysis App** – an interactive tool designed to help you analyze and manage risk in your portfolio. This Streamlit app provides comprehensive VaR calculations, stress testing, and dynamic NAV visualizations for Indian securities, giving you deep insights into your investments.

## Features

- **Dynamic VaR Calculations**  
  Compute both **Historical VaR** and **Parametric (Normal) VaR** for multiple securities with customizable confidence levels.

- **Stress Testing**  
  Evaluate your portfolio’s resilience with predefined stress scenarios (e.g., **COVID-19** and **Adani Crisis**) to understand potential downside risks.

- **Interactive NAV Visualizations**  
  Explore normalized NAV growth trends with interactive charts that feature a dynamic vertical cursor—hover to see detailed values for all securities at any given point in time.

- **Customizable Parameters**  
  Adjust the analysis period, investment amount, and confidence level effortlessly through an intuitive sidebar.

- **Automatic Data Fetching**  
  Historical data and ticker information (including full ETF names) are automatically retrieved from Yahoo Finance, ensuring you always work with the most up-to-date information.

## Hosted App

Experience the full functionality of the VaR Analysis App live at:  
[https://var-analysis.streamlit.app/](https://var-analysis.streamlit.app/)

## How It Works

1. **Data Acquisition**  
   The app downloads historical price data from Yahoo Finance.

2. **Risk Metrics Calculation**  
   - **Historical VaR:** Derived from the empirical distribution of historical returns.
   - **Parametric VaR:** Computed using a normal distribution model.
   - **Annual Volatility:** Adjusted for irregular trading frequencies for more precise risk estimates.
   - **Stress Tests:** Custom scenarios (e.g., COVID-19 and Adani Crisis) are applied individually to each security.

3. **Visualization**  
   An interactive NAV growth chart displays normalized prices (each series starting at 1) for all selected tickers. The chart features a dynamic vertical cursor that reveals the date and values for every fund as you hover over it.

## Installation

To run the app locally:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/var-analysis.git
   cd var-analysis
