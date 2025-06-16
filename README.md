# 🕸️ Graph_Viz - AI-Powered Graph Generator

Transform your ideas into stunning network visualizations with the power of AI! This Streamlit application uses Large Language Models to generate and visualize complex graphs from simple text prompts.

## ✨ Features

- **Natural Language to Graph**: Simply describe your network in plain English
- **AI-Powered Generation**: Leverages Google's Generative AI to understand and create graph structures
- **Multiple Layout Options**: Choose from various NetworkX and GraphViz layout algorithms
- **Interactive Visualization**: Built with NetworkX and Matplotlib for crisp, customizable graphs
- **Real-time Generation**: Instant graph creation and visualization
- **Export Ready**: Download your visualizations for presentations or reports

## 🚀 How It Works

1. **Enter Your Prompt**: Describe the network you want to visualize
   - "Create a social network of friends and their connections"
   - "Show the relationship between different programming languages"
   - "Generate a family tree structure"

2. **AI Processing**: The LLM analyzes your prompt and generates appropriate nodes and edges

3. **Instant Visualization**: Your graph appears with professional styling and layout

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Graph Processing**: NetworkX
- **AI Integration**: Google Generative AI (Gemini)
- **Visualization**: Matplotlib + PyDot
- **Layout Engine**: GraphViz

## 🎯 Use Cases

- **Educational**: Visualize concepts, relationships, and hierarchies
- **Business**: Create organizational charts, process flows, and system architectures
- **Research**: Generate network diagrams for academic papers and presentations
- **Creative**: Explore ideas through visual network representations

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8+
- Google AI API Key

### Local Installation

1. Clone the repository:
```bash
git clone https://github.com/balumantri46/Graph_viz.git
cd Graph_Viz
pip install -r requirements.txt
streamlit run app.py
