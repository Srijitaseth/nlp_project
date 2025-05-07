# nlp_project

This project provides a data-driven solution for optimizing **Quality of Service (QoS)** in live video game streaming platforms like **Twitch** and **YouTube Gaming**. The system focuses on analyzing real-time user feedback to enhance user experience by adjusting stream quality and buffering time based on engagement and satisfaction.

## Features

1. **BERT Sentiment Analysis**: Uses a pre-trained BERT model to classify sentiment from user reviews.
2. **DNN Satisfaction Prediction**: A custom Deep Neural Network (DNN) classifies user satisfaction based on review content.
3. **Real-Time QoS Adjustment**: Dynamically adjusts QoS based on sentiment and satisfaction classifications.

## Setup and Installation

### Prerequisites:
Ensure you have the following installed:
- Python 3.8 or higher
- Virtual Environment (optional but recommended)

### Steps:

1. **Clone the repository**:

    ```bash
    git clone https://github.com/your_username/qos_project.git
    cd qos_project
    ```

2. **Create and activate the virtual environment**:

    ```bash
    python -m venv qos_env
    source qos_env/bin/activate  # For Windows: qos_env\Scripts\activate
    ```

3. **Install the required dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. **Train and Test the Models**:
   - Run the following command to start the training process for both the **BERT** and **DNN models**:

    ```bash
    python src/main.py
    ```

   - This will train the models and save the best models as `sentiment_analysis_dnn_model.pth` and `sentiment_analysis_bert_model` for sentiment analysis.

2. **Start the FastAPI Server**:
   - To run the FastAPI server for real-time predictions:

    ```bash
    uvicorn src.api.api_server:app --reload
    ```

   - The API will be available at `http://127.0.0.1:8000`.

3. **Testing the API**:
   - You can interact with the **`/predict_qos/`** endpoint through the **Swagger UI** at `http://127.0.0.1:8000/docs` or use **Postman** to test the API by sending POST requests with user reviews.

### Sample Request to the API:

**POST** to `http://127.0.0.1:8000/predict_qos/`

```json
{
  "review": "The stream was lagging a bit, but the content was great!"
}
