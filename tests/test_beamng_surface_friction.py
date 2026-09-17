from src.simulation.beamng_surface_friction import BeamNGSurfaceFrictionQuery


class _LuaEndpoint:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def queue_lua_command(self, _chunk, response=False):
        assert response
        self.calls += 1
        return self.response


class _BeamNG:
    def __init__(self, response):
        self.control = _LuaEndpoint(response)


def test_queries_tractor_and_trailer_friction_separately():
    ground_models = [
        {
            "name": "ASPHALT",
            "collision_type": 1,
            "static_mu": 1.0,
            "sliding_mu": 0.8,
        },
        {
            "name": "ICE",
            "collision_type": 7,
            "static_mu": 0.2,
            "sliding_mu": 0.1,
        },
    ]
    tractor = _LuaEndpoint(
        {
            "effective_mu": 0.95,
            "contacts": [
                {"material_id": 1, "material": "Asphalt", "wheel_count": 4}
            ],
        }
    )
    trailer = _LuaEndpoint(
        {
            "effective_mu": 0.18,
            "contacts": [
                {"material_id": 7, "material": "Ice", "wheel_count": 2}
            ],
        }
    )

    result = BeamNGSurfaceFrictionQuery(
        _BeamNG(ground_models), tractor, trailer
    ).query()

    assert result.dynamics_mu == (0.95, 0.18)
    assert result.tractor.contacts[0].ground_models[0].name == "ASPHALT"
    assert result.trailer.contacts[0].ground_models[0].sliding_mu == 0.1


def test_invalid_or_airborne_friction_uses_fallback():
    result = BeamNGSurfaceFrictionQuery(
        _BeamNG([]),
        _LuaEndpoint({"effective_mu": 0.0, "contacts": []}),
        _LuaEndpoint({"contacts": []}),
        default_mu=0.7,
    ).query()

    assert result.dynamics_mu == (0.7, 0.7)
    assert result.tractor.contacts == ()


def test_ground_model_catalog_is_cached_until_refreshed():
    beamng = _BeamNG([])
    query = BeamNGSurfaceFrictionQuery(
        beamng,
        _LuaEndpoint({"effective_mu": 1.0, "contacts": []}),
        _LuaEndpoint({"effective_mu": 1.0, "contacts": []}),
    )

    query.query()
    query.query()
    assert beamng.control.calls == 1

    query.query(refresh_catalog=True)
    assert beamng.control.calls == 2


def test_beamng_string_responses_are_decoded():
    import json
    ground = [{"name": "ICE", "collision_type": 7, "static_mu": 0.2, "sliding_mu": 0.1}]
    contact = {"effective_mu": 0.18, "contacts": [{"material_id": 7, "material": "Ice", "wheel_count": 2}]}
    query = BeamNGSurfaceFrictionQuery(_BeamNG(json.dumps(ground)),
                                      _LuaEndpoint(json.dumps(contact)),
                                      _LuaEndpoint(json.dumps(contact)))
    result = query.query()
    assert result.dynamics_mu == (0.18, 0.18)
    assert result.tractor.contacts[0].ground_models[0].name == "ICE"
