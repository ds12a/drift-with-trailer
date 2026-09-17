"""Query BeamNG wheel contacts and turn them into dynamics friction inputs."""

from dataclasses import asdict, dataclass
import math
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class GroundModel:
    name: str
    collision_type: int
    static_mu: float
    sliding_mu: float


@dataclass(frozen=True, slots=True)
class ContactMaterial:
    material_id: int
    material: str
    wheel_count: int
    ground_models: tuple[GroundModel, ...]


@dataclass(frozen=True, slots=True)
class VehicleSurfaceFriction:
    effective_mu: float
    contacts: tuple[ContactMaterial, ...]


@dataclass(frozen=True, slots=True)
class RigSurfaceFriction:
    tractor: VehicleSurfaceFriction
    trailer: VehicleSurfaceFriction

    @property
    def dynamics_mu(self) -> tuple[float, float]:
        """Return ``(mu_tractor, mu_trailer)`` for the surface Fiala state."""
        return self.tractor.effective_mu, self.trailer.effective_mu

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_GROUND_MODELS_LUA = """
local result = {}
for name, entry in pairs(core_environment.groundModels) do
    local gm = entry.cdata
    table.insert(result, {
        name = name,
        collision_type = gm.collisiontype,
        static_mu = gm.staticFrictionCoefficient,
        sliding_mu = gm.slidingFrictionCoefficient
    })
end
return jsonEncode(result)
"""


_WHEEL_CONTACTS_LUA = """
local materials = require('particles').getMaterialsParticlesTable()
local counts = {}
for _, wheel in pairs(wheels.wheels) do
    local materialID = wheel.contactMaterialID1
    if materialID == nil or materialID < 0 or materialID == 4 then
        materialID = wheel.contactMaterialID2
    end
    -- Material 4 is the tire itself, not the contacted ground.
    if materialID ~= nil and materialID >= 0 and materialID ~= 4 then
        counts[materialID] = (counts[materialID] or 0) + 1
    end
end

local contacts = {}
for materialID, count in pairs(counts) do
    local material = materials[materialID]
    table.insert(contacts, {
        material_id = materialID,
        material = material and material.name or 'unknown',
        wheel_count = count
    })
end

return jsonEncode({
    effective_mu = obj:getStaticFrictionCoef(),
    contacts = contacts
})
"""


class BeamNGSurfaceFrictionQuery:
    """Read the surface independently under the tractor and trailer.

    BeamNG's ``obj:getStaticFrictionCoef()`` is used as the dynamics ``mu``.
    It is preferable to a material-name lookup because it is the effective
    coefficient seen by the current vehicle/tire/contact combination.  Wheel
    contact material IDs are still resolved against the ground-model catalog
    for logging and diagnostics.
    """

    def __init__(self, beamng, tractor, trailer, default_mu: float = 1.0):
        self.beamng = beamng
        self.tractor = tractor
        self.trailer = trailer
        self.default_mu = float(default_mu)
        self._catalog = None

    def query(self, refresh_catalog: bool = False) -> RigSurfaceFriction:
        if self._catalog is None or refresh_catalog:
            self._catalog = self._query_ground_models()
        return RigSurfaceFriction(
            tractor=self._query_vehicle(self.tractor, self._catalog),
            trailer=self._query_vehicle(self.trailer, self._catalog),
        )

    def _query_ground_models(self) -> dict[int, tuple[GroundModel, ...]]:
        response = self.beamng.control.queue_lua_command(
            _GROUND_MODELS_LUA, response=True
        ) or []
        if isinstance(response, str):
            response = json.loads(response)
        by_collision_type: dict[int, list[GroundModel]] = {}
        for raw in response:
            model = GroundModel(
                name=str(raw["name"]),
                collision_type=int(raw["collision_type"]),
                static_mu=float(raw["static_mu"]),
                sliding_mu=float(raw["sliding_mu"]),
            )
            by_collision_type.setdefault(model.collision_type, []).append(model)
        return {
            key: tuple(sorted(models, key=lambda model: model.name))
            for key, models in by_collision_type.items()
        }

    def _query_vehicle(
        self, vehicle, catalog: dict[int, tuple[GroundModel, ...]]
    ) -> VehicleSurfaceFriction:
        response = vehicle.queue_lua_command(_WHEEL_CONTACTS_LUA, response=True) or {}
        if isinstance(response, str):
            response = json.loads(response)
        raw_mu = response.get("effective_mu", self.default_mu)
        effective_mu = float(raw_mu) if raw_mu is not None else self.default_mu
        if not math.isfinite(effective_mu) or effective_mu <= 0:
            effective_mu = self.default_mu

        contacts = tuple(
            ContactMaterial(
                material_id=int(raw["material_id"]),
                material=str(raw["material"]),
                wheel_count=int(raw["wheel_count"]),
                ground_models=catalog.get(int(raw["material_id"]), ()),
            )
            for raw in sorted(
                response.get("contacts", []),
                key=lambda contact: int(contact["material_id"]),
            )
        )
        return VehicleSurfaceFriction(effective_mu=effective_mu, contacts=contacts)
