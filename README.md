# Dua Darling Lipa - Voice Chat Application

** The following is an AI generated README file for the Dua Darling Lipa project. **

Dua Darling Lipa is an interactive voice chat application that combines speech recognition, AI-driven conversational responses, and text-to-speech (TTS) capabilities. The app allows users to speak, receive AI-generated responses, and hear them in a synthesized voice.

## Features

- **Speech Recognition**: Converts user speech to text using browser-based speech recognition.
- **AI Chat Responses**: Generates conversational responses using a backend AI model.
- **Text-to-Speech (TTS)**: Streams audio responses in a synthesized voice.
- **Real-Time Streaming**: Uses Server-Sent Events (SSE) for real-time AI responses and TTS audio streaming.
- **Interactive UI**: Displays live transcripts, responses, and visual feedback for thinking and speaking states.

## Technologies Used

### Frontend
- React.js for the user interface.
- Browser-based Speech Recognition API.

### Backend
- FastAPI for the backend server.
- Google Cloud Speech-to-Text for transcription.
- Coqui TTS for text-to-speech synthesis.
- Ollama for AI conversational responses.
- SSE (Server-Sent Events) for real-time streaming.

## Setup Instructions

### Prerequisites
1. Node.js and npm installed for the frontend.
2. Python 3.8+ installed for the backend.
3. Google Cloud credentials for Speech-to-Text.
4. Install `ffmpeg` for audio processing.

### Frontend Setup
1. Navigate to the frontend directory:
   ```bash
   cd dua-darling-lipa/dua-chat-app
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Start the development server:
   ```bash
   npm start
   ```
   The app will be available at `http://localhost:3000`.

### Backend Setup
1. Navigate to the backend directory:
   ```bash
   cd dua-darling-lipa/backend
   ```
2. Create a `.env` file with the following variables:
   ```
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/google-cloud-credentials.json
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the backend server:
   ```bash
   uvicorn main:app --reload
   ```
   The backend will be available at `http://localhost:8000`.

### Running the Application
1. Start both the frontend and backend servers.
2. Open the frontend in your browser at `http://localhost:3000`.
3. Click the "Speak" button to start interacting with the app.

## Project Structure

```
dua-darling-lipa/
├── dua-chat-app/       # Frontend React application
├── backend/            # Backend FastAPI server
├── README.md           # Project documentation
```

## Future Enhancements
- Add support for multiple languages.
- Improve AI response personalization.
- Enhance UI/UX for better accessibility.

## License
This project is licensed under the MIT License.
