import React, { useState } from 'react'; import axios from 'axios'; import './App.css'; function App() { const [query, setQuery] = useState(''); const [image, setImage] = useState(null); const [response, setResponse] = useState(''); const [loading, setLoading] = useState(false); const [error, setError] = useState(''); const handleTextQuery = async () => { setLoading(true); setError(''); try { const res = await axios.post('http://tu-servidor.com/text_query', { query }); setResponse(res.data.response); } catch (err) { setError('Error al procesar la consulta: ' + err.message); } setLoading(false); }; const handleImageQuery = async () => { setLoading(true); setError(''); try { const formData = new FormData(); formData.append('file', image); formData.append('query', query); const res = await axios.post('http://tu-servidor.com/image_query', formData); setResponse(res.data.response); } catch (err) { setError('Error al procesar la imagen: ' + err.message); } setLoading(false); }; return ( <div className="App"> <h1>GrokClone Sin Restricciones</h1> <p>Pregunta cualquier cosa, sin límites. Usa con responsabilidad.</p> <textarea value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Pregunta lo que quieras..." rows="4" cols="50" /> <div> <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files[0])} /> </div> <div> <button onClick={handleTextQuery} disabled={loading}> {loading ? 'Procesando...' : 'Enviar Texto'} </button> <button onClick={handleImageQuery} disabled={loading || !image}> {loading ? 'Procesando...' : 'Enviar Imagen'} </button> </div> {error && <p className="error">{error}</p>} {response && ( <div className="response"> <h2>Respuesta:</h2> <p>{response}</p> </div> )} </div> ); } export default App;
