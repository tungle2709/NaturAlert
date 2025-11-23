import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './index.css';

window.__app_id = 'disaster-alert-demo';
window.__firebase_config = {
  apiKey: "AIzaSyDemoKey123",
  authDomain: "demo.firebaseapp.com",
  projectId: "demo-project",
  storageBucket: "demo.appspot.com",
  messagingSenderId: "123456789",
  appId: "1:123456789:web:abc123"
};
window.__initial_auth_token = undefined;

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
