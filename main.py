# Importing necessary libraries
from fastapi import FastAPI, APIRouter, Request, HTTPException
from runpod_serverless import ApiConfig, RunpodServerlessCompletion, Params, RunpodServerlessEmbedding
from fastapi.responses import StreamingResponse, JSONResponse
import json, time
from uvicorn import Config, Server
from pathlib import Path

# Initializing variables
model_data = {
    "object": "list",
    "data": []
}

configs = []
sdconfig = {} # SD webui config

def run(config_path: str, host: str = "127.0.0.1", port: int = 3000):
    global sdconfig

    if config_path:
        config_dict = load_config(config_path)  # function to load your config file

        for config in config_dict["models"]:
            if "type" in config and config["type"] == "sd_webui":
                sdconfig = config
                sdconfig["api_key"] = config_dict["api_key"]
                if "timeout" not in sdconfig:
                    sdconfig["timeout"] = 600
                continue

            config_model = {
                "url": f"https://api.runpod.ai/v2/{config['endpoint']}",
                "api_key": config_dict["api_key"],
                "model": config["model"],
                **({"timeout": config["timeout"]} if config.get("timeout") is not None else {}),
                **({"use_openai_format": config["use_openai_format"]} if config.get("use_openai_format") is not None else {}),
                **({"batch_size": config["batch_size"]} if config.get("batch_size") is not None else {}),
            }
            configs.append(ApiConfig(**config_model))
        for api in configs: print(api)

        model_data["data"] = [{"id": config["model"], 
            "object": "model", 
            "created": int(time.time()), 
            "owned_by": "organization-owner"} for config in config_dict["models"]]
        config = Config(
            app=app,
            host=config_dict.get("host", host),
            port=config_dict.get("port", port),
            log_level=config_dict.get("log_level", "info"),
        )
    else:
        config = Config(
            app=app,
            host=host,
            port=port,
            log_level="info",
        )
    server = Server(config=config)
    server.run()

def load_config(config_path):
    config_path = Path(args.config)
    with open(config_path) as f:
        return json.load(f)

# Function to get configuration by model name
def get_config_by_model(model_name):
    for config in configs:
        if config.model == model_name:
            return config
        
# Function to format the response data
def format_response(data):
    try:
        text_value = data['output'][0]['choices'][0]['tokens'][0]
    except (KeyError, IndexError, TypeError):
        try:
            text_value = data['output'][0]['choices'][0]['message']['content']
        except (KeyError, IndexError, TypeError):
            text_value = ''
    
    usage = data['output'][0].get('usage', {})

    if 'prompt_tokens' in usage and 'completion_tokens' in usage:
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
    elif 'input' in usage and 'output' in usage:
        prompt_tokens = usage.get('input', 0)
        completion_tokens = usage.get('output', 0)
        total_tokens = prompt_tokens + completion_tokens
    else:
        prompt_tokens = completion_tokens = total_tokens = 0

    openai_like_response = {
        'id': data['id'],
        'object': 'text_completion',
        'created': int(time.time()),
        'model': 'gpt-3.5-turbo-instruct',
        'system_fingerprint': "fp_44709d6fcb",
        'choices': [
            {
                'index': 0,
                'text': text_value,
                'logprobs': None,
                'finish_reason': 'stop' if data['status'] == 'COMPLETED' else 'length'
            }
        ],
        'usage': {
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens
        }
    }

    return openai_like_response

# Creating API router
router = APIRouter()

params = Params()

# API endpoint for chat completions
@router.post('/chat/completions')
async def request_chat(request: Request):
    try:
        data = await request.json()
        print(data)
        model = data.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"detail": "Missing model in request."})
        
        api = get_config_by_model(model)
        payload = data.get("messages")
        params_dict = params.dict()
        params_dict.update(data)
        new_params = Params(**params_dict)
        runpod: RunpodServerlessCompletion = RunpodServerlessCompletion(api=api, params=new_params)
        
        if(data["stream"]==False):
            response = get_chat_synchronous(runpod, payload)
            return response
        else:
            stream_data = get_chat_asynchronous(runpod, payload)
            response = StreamingResponse(content=stream_data, media_type="text/event-stream")
            response.body_iterator = stream_data.__aiter__()
            return response
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# API endpoint for completions
@router.post('/completions')
async def request_prompt(request: Request):
    try:
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"detail ": "Missing model in request."})
        payload = data.get("prompt")[0]
        api = get_config_by_model(model)
        
        params_dict = params.dict()
        params_dict.update(data)
        new_params = Params(**params_dict)
        runpod: RunpodServerlessCompletion = RunpodServerlessCompletion(api=api, params=new_params)
        return get_synchronous(runpod, payload)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# API endpoint for completions
@router.post('/embeddings')
async def request_embeddings(request: Request):
    try:
        data = await request.json()
        model = data.get("model")
        if not model:
            return JSONResponse(status_code=400, content={"detail ": "Missing model in request."})
        payload = data.get("input")
        api = get_config_by_model(model)
        runpod: RunpodServerlessEmbedding = RunpodServerlessEmbedding(api=api)
        return get_embedding(runpod, payload)
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

# Function to get chat synchronously
def get_chat_synchronous(runpod, chat):
    # Generate a response from the runpod
    response = runpod.generate(chat)
    # Check if the response is not cancelled
    if(response ['status'] != "CANCELLED"):
            # Extract data from the response
            data = response["output"][0]
    else:
        # If the request is cancelled, raise an exception
        raise HTTPException(status_code=408, detail="Request timed out.")
    return data

# Function to get chat asynchronously
async def get_chat_asynchronous(runpod, chat):
    # Generate a response from the runpod in an asynchronous manner
    async for chunk in runpod.stream_generate(chat):
        # print("STREAMING CHUNK=", type(chunk), chunk)
        # some events are dicts with status
        if isinstance(chunk, dict):

            if chunk.get("status") == "CANCELLED":
                raise HTTPException(status_code=408, detail="Request timed out.")

            stream = chunk.get("stream")

        else:
            # sometimes you get the raw list directly
            stream = chunk

        if not stream:
            continue

        # stream is always a list of {"output": "..."}
        for item in stream:
            output = item.get("output")
            if not output:
                continue

            # just forward exactly as received
            yield output.encode("utf-8")

# Function to get synchronous response
def get_synchronous(runpod, prompt):
    # Generate a response from the runpod
    response = runpod.generate(prompt)
    # Check if the response is not cancelled
    if(response['status'] != "CANCELLED"):
            # Format the response
            data = format_response(response)
    else:
        # If the request is cancelled, raise an exception
        raise HTTPException(status_code=408, detail="Request timed out.")
    return data

# Function to get synchronous response
def get_embedding(runpod, embedding):
    # Generate a response from the runpod
    response = runpod.generate(embedding)
    # Check if the response is not cancelled
    if(response['status'] != "CANCELLED"):
            # Format the response
            data = response["output"]
    else:
        # If the request is cancelled, raise an exception
        raise HTTPException(status_code=408, detail="Request timed out.")
    return data

# stable diffusion webui options
@router.get('/sdapi/v1/options')
async def request_sdoptions(request : Request):
    return {
        "jpeg_quality": 80,
        "webp_lossless": False,
    }

@router.get('/sdapi/v1/schedulers')
async def request_sdschedulers(request : Request):
    return [
        {
            "name": "automatic",
            "label": "Automatic",
            "aliases": None,
            "default_rho": -1,
            "need_inner_model": False
        }
    ]

@router.get('/sdapi/v1/samplers')
async def request_sdsamplers(request : Request):
    return [
        {
            "name": "Euler a",
            "aliases": [
                "k_euler_a",
                "k_euler_ancestral"
            ],
            "options": {
                "uses_ensd": "True"
            }
        }
    ]

@router.get('/sdapi/v1/sd-vae')
async def request_sdvae(request : Request):
    return []

@router.get('/sdapi/v1/sd-models')
async def request_sdmodels(request : Request):
    return [
        {
            "title": sdconfig["model"],
            "model": sdconfig["model"]
        }
    ]

@router.get('/sdapi/v1/upscalers')
async def request_sdupscalers():
    return []

@router.get('/sdapi/v1/latent-upscale-modes')
async def request_sd_latent_upscale_modes():
    return [
        {
            "name": "Latent"
        },
        {
            "name": "Latent (antialiased)"
        },
        {
            "name": "Latent (bicubic)"
        },
        {
            "name": "Latent (bicubic antialiased)"
        },
        {
            "name": "Latent (nearest)"
        },
        {
            "name": "Latent (nearest-exact)"
        }
    ]

@router.post('/sdapi/v1/options')
async def request_sd_set_options(request : Request):
    data = await request.json()
    print("SD Webui update options:", data)
    return JSONResponse(status_code=200, content={})

@router.post('/sdapi/v1/txt2img')
async def request_sd_txt2img(request : Request):
    import requests
    data = await request.json()
    print("SD webui txt2img: ", data)

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": sdconfig["api_key"],
    }
    runpodData = {
        "input": data
    }
    response = requests.post(f"https://api.runpod.ai/v2/{sdconfig["endpoint"]}/runsync", headers=headers, json=runpodData, 
                             timeout=sdconfig["timeout"])
    if (response.status_code != 200):
        return JSONResponse(status_code=response.status_code, content=response.json())
    
    json = response.json()
    if "status" in json and json["status"] != "COMPLETED":
        print(f"txt2img request status={json["status"]}!\n", json)
        return JSONResponse(status_code=500, content=json)
    
    if "output" not in json:
        print("txt2img request missing output!")
        return JSONResponse(status_code=500, content=json)
    
    print("Image generated, elapsed=",response.elapsed)
    # actual SD output
    output = json["output"]
    return JSONResponse(status_code=200, content=output)


# Create a FastAPI application
app = FastAPI()

# Include the router in the application
app.include_router(router)

# Endpoint to list all models
@app.get("/models")
async def list_models():
    return model_data

# Endpoint to get a specific model
@app.get("/models/{model_id}")
async def get_model(model_id):
    # Function to find a model by id
    def find_model(models, id):
        return next((model for model in models['data'] if model['id'] == id), None)
    # Return the found model
    return find_model(model_data, model_id)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Path to the config file", type=str, default=None)
    parser.add_argument("-e", "--endpoint", help="API endpoint", type=str, default=None)
    parser.add_argument("-k", "--api_key", help="API key", type=str, default=None)
    parser.add_argument("-m", "--model", help="Model", type=str, default=None)
    parser.add_argument("-t", "--timeout", help="Timeout", type=int, default=None)
    parser.add_argument("-o", "--use_openai_format", help="Use OpenAI format", type=bool, default=None)
    parser.add_argument("-b", "--batch_size", help="Batch size", type=int, default=None)
    parser.add_argument("--host", help="Host", type=str, default="127.0.0.1")
    parser.add_argument("--port", help="Port", type=int, default=3000)
    args = parser.parse_args()

    if args.config:
        run(args.config)
    else:
        config_model = {
            "url": f"https://api.runpod.ai/v2/{args.endpoint}",
            "api_key": args.api_key,
            "model": args.model,
            **({"timeout": args.timeout} if args.timeout is not None else {}),
            **({"use_openai_format": args.use_openai_format} if args.use_openai_format is not None else {}),
            **({"batch_size": args.batch_size} if args.batch_size is not None else {}),
        }
        configs.append(ApiConfig(**config_model))
        run(None, host=args.host, port=args.port)

