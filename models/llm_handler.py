import json
from typing import Optional, Dict, Any
from config.settings import settings

class LLMHandler:
    def __init__(self, provider: Optional[str] = None):
        self.provider = provider or settings.DEFAULT_LLM
        self.model: Any = None
        self.initialized = False
        
        if self.provider == "gemini":
            self._init_gemini()
        elif self.provider == "openai":
            self._init_openai()
    
    def _init_gemini(self) -> None:
        """Initialize Gemini model with error handling"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # List available models first
            available_models = []
            try:
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
                print(f"Available Gemini models: {available_models}")
            except:
                pass
            
            # Try different model names
            model_names = [
                'gemini-1.5-flash',  # Latest model
                'gemini-1.5-pro',    # Pro version
                'gemini-pro',        # Original
                'models/gemini-1.5-flash',  # With prefix
                'models/gemini-1.5-pro',
                'models/gemini-pro'
            ]
            
            for model_name in model_names:
                try:
                    self.model = genai.GenerativeModel(model_name)
                    # Test the model
                    test_response = self.model.generate_content("Hello")
                    if test_response:
                        print(f"Successfully initialized Gemini with model: {model_name}")
                        self.initialized = True
                        break
                except Exception as e:
                    print(f"Failed to initialize {model_name}: {e}")
                    continue
            
            if not self.initialized:
                print("Failed to initialize any Gemini model")
                
        except ImportError:
            print("Google Generative AI library not installed")
            self.initialized = False
        except Exception as e:
            print(f"Failed to initialize Gemini: {e}")
            self.initialized = False
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client with error handling"""
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.initialized = True
        except ImportError:
            try:
                import openai
                openai.api_key = settings.OPENAI_API_KEY
                self.openai_module = openai
                self.initialized = True
                self.use_old_api = True
            except Exception as e:
                print(f"Failed to initialize OpenAI: {e}")
                self.initialized = False
        except Exception as e:
            print(f"Failed to initialize OpenAI: {e}")
            self.initialized = False
            
    def generate_response(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """Generate response from LLM with error handling"""
        if not prompt:
            return None
            
        if not self.initialized:
            return f"LLM not initialized. Please check your API keys and model availability."
            
        try:
            if self.provider == "gemini":
                return self._generate_gemini(prompt, temperature)
            elif self.provider == "openai":
                return self._generate_openai(prompt, temperature)
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return None
    
    def _generate_gemini(self, prompt: str, temperature: float) -> Optional[str]:
        """Generate response using Gemini"""
        if not self.model:
            return None
            
        try:
            # Try with generation config
            generation_config = {
                'temperature': temperature,
                'top_p': 0.95,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
            
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            except:
                # Fallback to simple generation
                response = self.model.generate_content(prompt)
            
            # Extract text from response
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'parts') and response.parts:
                text_parts = []
                for part in response.parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                return ' '.join(text_parts) if text_parts else None
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    text_parts = []
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            text_parts.append(part.text)
                    return ' '.join(text_parts) if text_parts else None
            
            # Last resort - convert to string
            return str(response) if response else None
            
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return None
    
    def _generate_openai(self, prompt: str, temperature: float) -> Optional[str]:
        """Generate response using OpenAI"""
        try:
            if hasattr(self, 'client'):
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2048
                )
                if response and response.choices:
                    return response.choices[0].message.content
            elif hasattr(self, 'openai_module'):
                response = self.openai_module.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=2048
                )
                if response and 'choices' in response:
                    return response['choices'][0]['message']['content']
            
            return None
            
        except Exception as e:
            print(f"OpenAI generation error: {e}")
            return None
    
    def parse_json_response(self, response: Optional[str]) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response with error handling"""
        if not response:
            return None
            
        try:
            # Clean the response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if '```json' in response:
                start = response.find('```json') + 7
                end = response.find('```', start)
                if end > start:
                    response = response[start:end].strip()
            elif '```' in response:
                start = response.find('```') + 3
                end = response.find('```', start)
                if end > start:
                    response = response[start:end].strip()
            
            # Try direct parsing
            try:
                return json.loads(response)
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON object
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Try to extract JSON array
            start_idx = response.find('[')
            end_idx = response.rfind(']')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            return None
            
        except Exception as e:
            print(f"Error parsing JSON from response: {e}")
            return None