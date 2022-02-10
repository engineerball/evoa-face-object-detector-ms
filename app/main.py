import os
import io
import uvicorn
import sys

from fastapi import FastAPI, HTTPException, Body, status
from fastapi.responses import Response, JSONResponse
from loguru import logger
from pydantic import BaseModel

from app.objectdetector import load_interpreter, predict
from app.utils import stringToRGB, get_rule_from_file
from app.utils import ALLOW_LIST, DENY_LIST

import app.settings as ste

get_rule_from_file(ALLOW_LIST, ste.ALLOW_LIST_FILE)
get_rule_from_file(DENY_LIST, ste.DENY_LIST_FILE)

app = FastAPI(title='EVOA Face Object Detector')

FACE_OBJECT_DETECTOR_ENDPOINT = "/v1/evoa-face/verify"

# TENSORFLOW_URL = ste.TENSORFLOW_URL
INTERPRETER = load_interpreter()

class Item(BaseModel):
    encoded_img: str

class Detector(BaseModel):
    allowed: bool
    
class Response(BaseModel):
    result: Detector
    status_code: int
    status_message: str
    
    
@app.get("/")
def home():
    return "API is working as expected."


@app.post(FACE_OBJECT_DETECTOR_ENDPOINT, response_model=Response)
async def prediction(req_body: Item = Body(...)):
    try:
        # image = stringToRGB(req_body.encoded_img)
    
        # result = predict(image)
        logger.debug(INTERPRETER.get_signature_list())
        result = predict(req_body.encoded_img, INTERPRETER)
        
        if result:
            output = {
                    "result": result,
                    "status_code": 200,
                    "status_message": "OK"
            }
                # if return_mrz_image:
                #     output['result']['mrz_image'] =  cv_to_base64(mrz_image)
            status_code = status.HTTP_200_OK
        else:
            output = {
                "result": result,
                "status_code": 422,
                "status_message": "Something went wrong"
            }
            status_code = status.HTTP_422_UNPROCESSABLE_ENTITY    
        return JSONResponse(content=output, status_code=status_code)
    except:
        e = sys.exc_info()[1]
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

