// src/App.jsx

import React, { useState, useEffect, useRef } from 'react';
import './App.css';


function App() {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [error, setError] = useState('');
  
  const recognitionRef = useRef(null);
  const audioRef = useRef(new Audio());
  
  // Initialize speech recognition
  useEffect(() => {
    // Check if SpeechRecognition is available
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
      setError('Speech recognition not supported in this browser.');
      return;
    }
    
    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    recognition.onstart = () => {
      console.log('Speech recognition started');
    };
    
    recognition.onresult = (event) => {
      const transcriptResult = Array.from(event.results)
        .map(result => result[0])
        .map(result => result.transcript)
        .join('');
      
      setTranscript(transcriptResult);
    };
    
    recognition.onend = () => {
      console.log('Speech recognition ended');
      setIsListening(false);
    };
    
    recognition.onerror = (event) => {
      console.error('Speech recognition error', event.error);
      setError(`Speech recognition error: ${event.error}`);
      setIsListening(false);
    };
    
    recognitionRef.current = recognition;
    
    // Cleanup
    return () => {
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch (e) {
          console.error('Error stopping speech recognition:', e);
        }
      }
    };
  }, []);
  
  // Function to toggle listening state
  const toggleListening = () => {
    if (isListening) {
      try {
        recognitionRef.current.stop();
        // Process what was said
        if (transcript.trim()) {
          handleSubmit();
        }
      } catch (e) {
        console.error('Error stopping recognition:', e);
        setError(`Error stopping microphone: ${e.message}`);
      }
    } else {
      setError('');
      setTranscript('');
      setResponse('');
      
      try {
        recognitionRef.current.start();
        setIsListening(true);
      } catch (e) {
        console.error('Error starting recognition:', e);
        setError(`Error starting microphone: ${e.message}`);
      }
    }
  };
  
  // Handle form submission
  const handleSubmit = async () => {
    if (!transcript.trim()) return;
    
    setIsThinking(true);
    
    try {
      // First, get AI response via SSE
      const resp = await streamChat(transcript);
      setResponse(resp);
      
      // Then, stream TTS audio
      if (resp) {
        setIsSpeaking(true);
        await playAudio(resp);
      }
    } catch (error) {
      console.error('Error processing request:', error);
      setError(`Error processing your request: ${error.message}`);
    } finally {
      setIsThinking(false);
      setIsSpeaking(false);
    }
  };
  
  // Stream chat with SSE
  const streamChat = (prompt) => {
    return new Promise((resolve, reject) => {
      try {
        const encodedPrompt = encodeURIComponent("keep your response short to this:"+prompt);
        const eventSource = new EventSource(`http://localhost:8000/stream_chat/?prompt=${encodedPrompt}`);
        
        let fullResponse = '';
        
        eventSource.onopen = () => {
          console.log('SSE connection opened');
        };
        
        eventSource.addEventListener('message', (event) => {
          fullResponse += event.data;
          setResponse(fullResponse);
        });
        
        eventSource.addEventListener('done', () => {
          console.log('SSE stream complete');
          eventSource.close();
          resolve(fullResponse);
        });
        
        eventSource.onerror = (event) => {
          console.error('SSE error:', event);
          eventSource.close();
          reject(new Error('Error streaming response from server'));
        };
      } catch (err) {
        reject(err);
      }
    });
  };
  
  // Play TTS audio
  const playAudio = async (text) => {
    return new Promise((resolve, reject) => {
      try {
        const encodedText = encodeURIComponent(text);
        const audioUrl = `http://localhost:8000/stream_tts/?text=${encodedText}`;
        
        console.log('Setting up audio with URL:', audioUrl);
        
        // Create an audio context instead of using the Audio API
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        const audioContext = new AudioContext();
        
        console.log('Fetching audio data from server...');
        
        // Fetch the audio data as an ArrayBuffer
        fetch(audioUrl)
          .then(response => {
            if (!response.ok) {
              throw new Error(`HTTP error! Status: ${response.status}`);
            }
            console.log('Response headers:', response.headers);
            return response.arrayBuffer();
          })
          .then(arrayBuffer => {
            console.log('Audio data received, size:', arrayBuffer.byteLength, 'bytes');
            if (arrayBuffer.byteLength === 0) {
              throw new Error('Received empty audio data');
            }
            
            // Set speaking state
            setIsSpeaking(true);
            
            // Decode the audio data
            return audioContext.decodeAudioData(arrayBuffer);
          })
          .then(audioBuffer => {
            console.log('Audio decoded successfully, duration:', audioBuffer.duration, 's');
            
            // Create a buffer source
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            
            // Handle completion
            source.onended = () => {
              console.log('Audio playback complete');
              setIsSpeaking(false);
              resolve();
            };
            
            // Start playback
            console.log('Starting audio playback');
            source.start(0);
          })
          .catch(error => {
            console.error('Error with audio processing:', error);
            setIsSpeaking(false);
            reject(error);
          });
      } catch (err) {
        console.error('Error setting up audio context:', err);
        setIsSpeaking(false);
        reject(err);
      }
    });
  };
  
  return (
    <div className="app">
      <div className="dua-container">
        <div className={`dua-dot ${isThinking ? 'thinking' : ''} ${isSpeaking ? 'speaking' : ''}`}>
          <div className="glow"></div>
        </div>
      </div>
      
      <div className="input-area">
        {error && <p className="error">{error}</p>}
        <p className="transcript">{transcript}</p>
        <p className="response">{response}</p>
        
        <button 
          className={`mic-button ${isListening ? 'active' : ''}`} 
          onClick={toggleListening}
        >
          {isListening ? 'Stop' : 'Speak'}
        </button>
      </div>
    </div>
  );
}

export default App;