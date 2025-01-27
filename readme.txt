
## Prerequisites
1. **Python 3.11 or later** installed on your local machine.
2. **Docker** installed for containerization.
3. A trained ML model (`hdbscan_face_recognition_model.pkl`) stored in the project directory.

## Installation Steps

### 1. Set Up the Environment
   - Clone or download the repository to your local machine.
   - Navigate to the project directory:
     ```
     cd /path/to/project-directory
     ```

### 2. Install Python Dependencies
   - If running locally (outside of Docker), install the required Python packages:
     ```
     pip install -r requirements.txt
     ```

### 3. Build the Docker Image
   - To containerize the application, run:
     ```
     docker build -t face-recognition-app .
     ```

### 4. Run the Docker Container
   - Deploy the application on port 5000 using the following command:
     ```
     docker run -d -p 5000:5000 --name face-recognition face-recognition-app
     ```

   - Verify that the application is running by navigating to `http://localhost:5000` in your web browser.

## Project Features
1. **Face Recognition with HDBSCAN**:
   - The model processes input data and predicts cluster assignments for face recognition.
2. **Flask API**:
   - A RESTful API is exposed to handle incoming data and return results in JSON format.
3. **Docker Deployment**:
   - The entire application is containerized for portability and easy deployment.

## Usage
1. **Upload Input Data**: Send a POST request to the Flask API endpoint with input data.
2. **Get Predictions**: The application returns cluster predictions for the uploaded data.

### Sample API Request (Using cURL)
```bash
curl -X POST -H "Content-Type: application/json" -d '{"data": [features]}' http://localhost:5000/predict
