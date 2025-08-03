Smart Farming AI Agent 🌾

A simple AI-powered web application designed to assist small-scale farmers with real-time, localized crop recommendations based on soil and weather data.


Project Overview

This project leverages machine learning to analyze key agricultural features such as soil nutrients (Nitrogen, Phosphorus, Potassium), temperature, humidity, pH, and rainfall. It predicts the most suitable crop for the current season, helping farmers reduce risk, increase crop yield, and optimize income.

The application includes an easy-to-use Flask-based web interface where farmers can input their field data and receive instant crop recommendations.


Features

- Crop recommendation using a LightGBM machine learning model.
- Input form for soil and weather parameters in the web interface.
- Model training script included (`train_model.py`) to train and save the ML model.
- Simple and intuitive UI for non-technical users.
- Easily extensible to integrate real-time weather data, market prices, and RAG technologies using IBM Cloud/Granite.
