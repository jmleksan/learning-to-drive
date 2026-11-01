"""Minimal OpenDRIVE literacy sample (offline; not a full Town01 map)."""

# Tiny synthetic OpenDRIVE-like snippet for schema inspection without CARLA.
SAMPLE_XODR = """<?xml version="1.0" standalone="yes"?>
<OpenDRIVE>
  <header revMajor="1" revMinor="4" name="sample_corridor" version="1.0"
          date="2026-10-19" north="50" south="0" east="200" west="0"/>
  <road name="main" length="200.0" id="1" junction="-1">
    <link/>
    <planView>
      <geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="200.0">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0.0">
        <left>
          <lane id="1" type="driving" level="true">
            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </left>
        <center>
          <lane id="0" type="none" level="true"/>
        </center>
        <right>
          <lane id="-1" type="driving" level="true">
            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
  <road name="side" length="80.0" id="2" junction="10">
    <link>
      <predecessor elementType="road" elementId="1" contactPoint="end"/>
    </link>
    <planView>
      <geometry s="0.0" x="200.0" y="0.0" hdg="1.57" length="80.0">
        <line/>
      </geometry>
    </planView>
    <lanes>
      <laneSection s="0.0">
        <center><lane id="0" type="none" level="true"/></center>
        <right>
          <lane id="-1" type="driving" level="true">
            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
  </road>
  <junction id="10" name="T">
    <connection id="0" incomingRoad="1" connectingRoad="2" contactPoint="start"/>
  </junction>
</OpenDRIVE>
"""


def parse_opendrive_stats(xodr: str) -> dict:
    """Lightweight XML counts for teaching (not a full OpenDRIVE parser)."""
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xodr)
    roads = root.findall("road")
    junctions = root.findall("junction")
    lanes = root.findall(".//lane")
    driving = [ln for ln in lanes if ln.get("type") == "driving"]
    return {
        "n_roads": len(roads),
        "n_junctions": len(junctions),
        "n_lane_elements": len(lanes),
        "n_driving_lanes": len(driving),
        "road_ids": [r.get("id") for r in roads],
        "source": "offline_sample",
    }
