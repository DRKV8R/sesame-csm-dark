import React, { useState, useRef, useCallback } from 'react';
import { Upload, Play, Download, Loader, AlertCircle, CheckCircle, Video } from 'lucide-react';

const VideoGenerationTab = () => {
  const [uploadedImages, setUploadedImages] = useState([]);
  const [characterName, setCharacterName] = useState('');
  const [loraTraining, setLoraTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState(0);
  const [trainedLoraId, setTrainedLoraId] = useState('');
  
  const [videoPrompt, setVideoPrompt] = useState('');
  const [generatingVideo, setGeneratingVideo] = useState(false);
  const [generatedVideo, setGeneratedVideo] = useState('');
  const [videoDuration, setVideoDuration] = useState(4);
  
  const [availableLoras, setAvailableLoras] = useState([]);
  const [selectedLora, setSelectedLora] = useState('');
  
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);

  // Mai's example prompts for demonstration
  const examplePrompts = [
    "Professional woman in business attire walking through modern office hallway",
    "Elegant woman sitting at conference table presenting documents to camera",
    "Professional assistant organizing files in bright office environment",
    "Business woman greeting visitors in corporate reception area",
    "Professional woman explaining charts and graphs in meeting room"
  ];

  const handleImageUpload = useCallback((event) => {
    const files = Array.from(event.target.files);
    
    files.forEach(file => {
      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const imageData = {
            id: Date.now() + Math.random(),
            name: file.name,
            base64: e.target.result.split(',')[1], // Remove data:image/jpeg;base64, prefix
            preview: e.target.result
          };
          setUploadedImages(prev => [...prev, imageData]);
        };
        reader.readAsDataURL(file);
      }
    });
  }, []);

  const removeImage = useCallback((imageId) => {
    setUploadedImages(prev => prev.filter(img => img.id !== imageId));
  }, []);

  const trainLoraModel = async () => {
    if (uploadedImages.length < 3) {
      alert('Please upload at least 3 images for LoRA training');
      return;
    }
    
    if (!characterName.trim()) {
      alert('Please enter a character name');
      return;
    }

    setLoraTraining(true);
    setTrainingProgress(0);

    try {
      const response = await fetch(`https://api.runpod.ai/v2/${process.env.NEXT_PUBLIC_WAN_ENDPOINT_ID}/runsync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.NEXT_PUBLIC_RUNPOD_API_KEY}`
        },
        body: JSON.stringify({
          input: {
            action: 'train_lora',
            images: uploadedImages.map(img => img.base64),
            character_name: characterName
          }
        })
      });

      const result = await response.json();

      if (result.output?.status === 'success') {
        setTrainedLoraId(result.output.lora_id);
        setAvailableLoras(prev => [...prev, {
          id: result.output.lora_id,
          name: characterName,
          created: new Date().toLocaleDateString()
        }]);
        alert(`LoRA training completed for ${characterName}!`);
      } else {
        throw new Error(result.output?.error || 'Training failed');
      }
    } catch (error) {
      console.error('LoRA training error:', error);
      alert(`Training failed: ${error.message}`);
    } finally {
      setLoraTraining(false);
      setTrainingProgress(0);
    }
  };

  const generateVideo = async () => {
    if (!videoPrompt.trim()) {
      alert('Please enter a video prompt');
      return;
    }

    setGeneratingVideo(true);

    try {
      const response = await fetch(`https://api.runpod.ai/v2/${process.env.NEXT_PUBLIC_WAN_ENDPOINT_ID}/runsync`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${process.env.NEXT_PUBLIC_RUNPOD_API_KEY}`
        },
        body: JSON.stringify({
          input: {
            action: 'generate',
            prompt: videoPrompt,
            character_lora: selectedLora || trainedLoraId || null,
            duration: videoDuration
          }
        })
      });

      const result = await response.json();

      if (result.output?.video_base64) {
        const videoBlob = new Blob([
          Uint8Array.from(atob(result.output.video_base64), c => c.charCodeAt(0))
        ], { type: 'video/mp4' });
        
        const videoUrl = URL.createObjectURL(videoBlob);
        setGeneratedVideo(videoUrl);
      } else {
        throw new Error(result.output?.error || 'Video generation failed');
      }
    } catch (error) {
      console.error('Video generation error:', error);
      alert(`Generation failed: ${error.message}`);
    } finally {
      setGeneratingVideo(false);
    }
  };

  const downloadVideo = () => {
    if (generatedVideo) {
      const link = document.createElement('a');
      link.href = generatedVideo;
      link.download = `mai_video_${Date.now()}.mp4`;
      link.click();
    }
  };

  return (
    <div className="space-y-8 p-6">
      {/* Header */}
      <div className="text-center">
        <h2 className="text-3xl font-bold text-white mb-4">
          🎬 WAN 2.1 Video Generation
        </h2>
        <p className="text-gray-300 max-w-2xl mx-auto">
          Create professional videos with AI. Train custom character models and generate 
          consistent videos for your portfolio presentations.
        </p>
      </div>

      {/* Character Training Section */}
      <div className="glass-card rounded-2xl p-8">
        <h3 className="text-xl font-bold text-white mb-6 flex items-center">
          <Upload className="w-6 h-6 mr-3 text-blue-400" />
          Train Character LoRA
        </h3>

        {/* Image Upload */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-3">
            Upload Character Images (minimum 3 required)
          </label>
          
          <div 
            className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center cursor-pointer hover:border-blue-400 transition-colors"
            onClick={() => fileInputRef.current?.click()}
          >
            <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <p className="text-gray-300">Click to upload images or drag and drop</p>
            <p className="text-sm text-gray-500 mt-2">
              JPG, PNG, WebP • High quality portraits work best
            </p>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />
        </div>

        {/* Uploaded Images Preview */}
        {uploadedImages.length > 0 && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-3">
              Uploaded Images ({uploadedImages.length})
            </label>
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              {uploadedImages.map(image => (
                <div key={image.id} className="relative group">
                  <img 
                    src={image.preview} 
                    alt={image.name}
                    className="w-full h-24 object-cover rounded-lg"
                  />
                  <button
                    onClick={() => removeImage(image.id)}
                    className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs hover:bg-red-600 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Character Name */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Character Name
          </label>
          <input
            type="text"
            value={characterName}
            onChange={(e) => setCharacterName(e.target.value)}
            placeholder="Enter character name (e.g., Mai)"
            className="w-full bg-black/50 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:border-blue-400 focus:outline-none"
          />
        </div>

        {/* Train Button */}
        <button
          onClick={trainLoraModel}
          disabled={loraTraining || uploadedImages.length < 3 || !characterName.trim()}
          className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white py-3 px-6 rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:from-blue-600 hover:to-purple-700 transition-all duration-300 flex items-center justify-center"
        >
          {loraTraining ? (
            <>
              <Loader className="w-5 h-5 mr-2 animate-spin" />
              Training LoRA... {trainingProgress}%
            </>
          ) : (
            <>
              <CheckCircle className="w-5 h-5 mr-2" />
              Train Character Model
            </>
          )}
        </button>

        {trainedLoraId && (
          <div className="mt-4 p-4 bg-green-500/20 border border-green-500/30 rounded-lg">
            <p className="text-green-400 flex items-center">
              <CheckCircle className="w-5 h-5 mr-2" />
              LoRA training completed! Character "{characterName}" is ready for video generation.
            </p>
          </div>
        )}
      </div>

      {/* Video Generation Section */}
      <div className="glass-card rounded-2xl p-8">
        <h3 className="text-xl font-bold text-white mb-6 flex items-center">
          <Video className="w-6 h-6 mr-3 text-red-400" />
          Generate Video
        </h3>

        {/* LoRA Selection */}
        {(availableLoras.length > 0 || trainedLoraId) && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Character Model (Optional)
            </label>
            <select
              value={selectedLora}
              onChange={(e) => setSelectedLora(e.target.value)}
              className="w-full bg-black/50 border border-gray-600 rounded-lg px-4 py-3 text-white focus:border-blue-400 focus:outline-none"
            >
              <option value="">Default (no character model)</option>
              {trainedLoraId && (
                <option value={trainedLoraId}>
                  {characterName} (just trained)
                </option>
              )}
              {availableLoras.map(lora => (
                <option key={lora.id} value={lora.id}>
                  {lora.name} (created {lora.created})
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Video Prompt */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Video Description
          </label>
          <textarea
            value={videoPrompt}
            onChange={(e) => setVideoPrompt(e.target.value)}
            placeholder="Describe the video you want to generate..."
            rows={4}
            className="w-full bg-black/50 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:border-blue-400 focus:outline-none resize-none"
          />
        </div>

        {/* Example Prompts */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Example Prompts (Mai Assistant)
          </label>
          <div className="flex flex-wrap gap-2">
            {examplePrompts.map((prompt, index) => (
              <button
                key={index}
                onClick={() => setVideoPrompt(prompt)}
                className="text-xs bg-gray-700 hover:bg-gray-600 text-gray-300 px-3 py-1 rounded-full transition-colors"
              >
                {prompt.slice(0, 50)}...
              </button>
            ))}
          </div>
        </div>

        {/* Duration Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Duration: {videoDuration} seconds
          </label>
          <input
            type="range"
            min={2}
            max={6}
            value={videoDuration}
            onChange={(e) => setVideoDuration(parseInt(e.target.value))}
            className="w-full"
          />
        </div>

        {/* Generate Button */}
        <button
          onClick={generateVideo}
          disabled={generatingVideo || !videoPrompt.trim()}
          className="w-full bg-gradient-to-r from-red-500 to-pink-600 text-white py-3 px-6 rounded-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed hover:from-red-600 hover:to-pink-700 transition-all duration-300 flex items-center justify-center"
        >
          {generatingVideo ? (
            <>
              <Loader className="w-5 h-5 mr-2 animate-spin" />
              Generating Video...
            </>
          ) : (
            <>
              <Play className="w-5 h-5 mr-2" />
              Generate Video
            </>
          )}
        </button>
      </div>

      {/* Generated Video Preview */}
      {generatedVideo && (
        <div className="glass-card rounded-2xl p-8">
          <h3 className="text-xl font-bold text-white mb-6">Generated Video</h3>
          
          <div className="mb-6">
            <video
              ref={videoRef}
              src={generatedVideo}
              controls
              className="w-full max-w-2xl mx-auto rounded-lg"
            />
          </div>

          <div className="flex justify-center space-x-4">
            <button
              onClick={() => videoRef.current?.play()}
              className="bg-green-500 hover:bg-green-600 text-white px-6 py-2 rounded-lg flex items-center transition-colors"
            >
              <Play className="w-4 h-4 mr-2" />
              Play
            </button>
            <button
              onClick={downloadVideo}
              className="bg-blue-500 hover:bg-blue-600 text-white px-6 py-2 rounded-lg flex items-center transition-colors"
            >
              <Download className="w-4 h-4 mr-2" />
              Download
            </button>
          </div>
        </div>
      )}

      {/* Usage Info */}
      <div className="glass-card rounded-2xl p-6">
        <div className="flex items-center mb-4">
          <AlertCircle className="w-5 h-5 text-yellow-400 mr-2" />
          <h4 className="text-lg font-semibold text-white">Usage & Costs</h4>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <strong className="text-green-400">LoRA Training:</strong>
            <p className="text-gray-300">~$0.50 per character model</p>
          </div>
          <div>
            <strong className="text-blue-400">Video Generation:</strong>
            <p className="text-gray-300">~$0.10 per 4-second video</p>
          </div>
          <div>
            <strong className="text-purple-400">Idle Cost:</strong>
            <p className="text-gray-300">$0.000 when not generating</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoGenerationTab;
