from fastapi import APIRouter

from rrp_service.py import RrpService

router = APIRouter(prefix="/rrp")

@router.post("/")
def get_reinspection_risk():
    return RrpService.predict_reinspection_risk()
    
@router.post("/graph")
def get_rrp_graph():
    return RrpService.generate_rrp_graph()