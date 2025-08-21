import logo from './logo.svg';
import './App.css';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import ChatBot from './components/chatbot';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<ChatBot />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
