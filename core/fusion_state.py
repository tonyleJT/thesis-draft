# MetroPaper/core/fusion_state.py

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from MetroPaper.config import (
    PHASE_ENTRY,
    PHASE_AFTER_ESCALATOR_1,
    PHASE_AFTER_TICKET,
    PHASE_AFTER_ESCALATOR_2,
)

from MetroPaper.models.od_model import ODResult, ODDetection
from MetroPaper.models.ss_model import SSResult
from MetroPaper.models.ocr_model import OCRResult, OCRPair
from MetroPaper.utils.speaker import Speaker
from MetroPaper.utils.clock_sync import points_to_clock


@dataclass
class GuidanceDecision:
    # what to draw
    target_pt: Optional[Tuple[int, int]] = None
    # what to show (persist until next spoken message replaces it)
    ui_text: str = ""
    # for debugging / consistency if needed
    clock_dir: Optional[str] = None
    angle_deg: float = 0.0


class NavigationFusion:
    """
    Rebuilt fusion:
    - Models run normally.
    - Only show/speak what blind should follow (no internal debug).
    - UI text persists until next spoken message replaces it.
    """

    # user-requested cadence ~3–5s
    GUIDE_COOLDOWN_SEC = 4.0

    def __init__(self, speaker: Speaker):
        self.speaker = speaker

        self.phase = PHASE_ENTRY
        self.ss_enabled = True

        # escalator visibility tracking -> phase transitions when escalator disappears
        self._escalator_was_visible = False

        # ticket booth timing for stage transition
        self.last_ticket_booth_time = 0.0
        self.saw_ticket_booth = False

        # repeating global info (stage3)
        self.last_ben_thanh_repeat = 0.0

        # speech de-dup (by full sentence)
        self.last_guidance_speak_time = 0.0
        self.last_guidance_text = ""

        # UI display state: stays until NEXT spoken message replaces it
        self.ui_text = ""

        # stage flags
        self.flags = {}

        # stage2 special
        self.stage2_turnleft_done = False
        self.stage2_suppress_ss_until_gate_ahead = False
        self.stage2_straight_only_until_ticketbooth = False

        # Stage 2 safety: do not allow "Turn left" immediately after gate detection
        self.stage2_gate_seen_time = 0.0
        self.stage2_gate_direction_known = False
        self.STAGE2_TURNLEFT_ARM_DELAY_SEC = 2.0

        # == This is new: scene phase inference for robustness to cut sections ==
        self._candidate_phase = self.phase
        self._candidate_phase_count = 0
        self.PHASE_CONFIRM_FRAMES = 3

    # ---------- helpers ----------

    def _set_once(self, key: str) -> bool:
        if self.flags.get(key, False):
            return False
        self.flags[key] = True
        return True

    def _find_pair(self, ocr_res: OCRResult, type_name: str) -> Optional[OCRPair]:
        for p in ocr_res.platforms:
            for pair in p.pairs:
                if pair.type == type_name:
                    return pair
        return None

    def _say_and_display(self, text: str, force: bool = False):
        """
        Speak + set on-screen text.
        Text persists until another spoken message replaces it.
        """
        if not text:
            return
        self.speaker.say(text, force=force)
        self.ui_text = text

    def _cooldown_guidance(self, now: float, new_text: str) -> bool:
        """
        Every GUIDE_COOLDOWN_SEC seconds:
        - If new_text != last spoken -> allow speaking (and update display)
        - If same -> no speaking, display remains unchanged
        """
        if now - self.last_guidance_speak_time < self.GUIDE_COOLDOWN_SEC:
            return False
        if new_text == self.last_guidance_text:
            return False
        return True

    def _commit_guidance(self, now: float, text: str, force: bool = False):
        self._say_and_display(text, force=force)
        self.last_guidance_speak_time = now
        self.last_guidance_text = text

    def _od_best(self, od: ODResult, class_name: str) -> Optional[ODDetection]:
        best = None
        for d in od.detections:
            if d.class_name == class_name:
                if (best is None) or (d.conf > best.conf):
                    best = d
        return best

    def _apply_filter(
        self,
        root_xy: Tuple[int, int],
        target_pt: Tuple[int, int],
    ) -> Tuple[bool, Optional[str], float]:
        """
        Returns (ok, clock, angle_deg) with current phase filters:
        - stage2 straight_only: abs(angle)<=15 deg
        - stage3/4 left_only: angle<=0 (left or straight)
        """
        clock, angle = points_to_clock(root_xy, target_pt)
        if clock is None:
            return False, None, angle

        # stage2 straight-only
        if self.stage2_straight_only_until_ticketbooth:
            if abs(angle) > 15.0:
                return False, clock, angle

        # stage3/4 left-only
        if self.phase in (PHASE_AFTER_TICKET, PHASE_AFTER_ESCALATOR_2):
            if angle > 0.0:
                return False, clock, angle

        return True, clock, angle


    # == This is new !
    def _infer_scene_phase(self, od: ODResult, ss: SSResult, ocr: OCRResult) -> int:
        """
        Infer navigation phase from current visible evidence.
        This makes the system robust when starting from a cut video section
        instead of always starting from the entrance.
        """

        # OD landmarks
        stair = self._od_best(od, "stair node")
        ticket_booth = self._od_best(od, "ticket booth")

        # OCR semantic landmarks
        pair_noentry_gate = self._find_pair(ocr, "no-entry_gate")
        pair_ben_suoi = self._find_pair(ocr, "ben-thanh_suoi-tien")

        has_platform_sign = len(ocr.platforms) > 0
        has_ben_in_platform = any(
            c.class_name == "ben-thanh-station"
            for p in ocr.platforms
            for c in p.components
        )

        # ---------------- Highest-confidence scene inference ----------------

        # Final platform/train area:
        # platform sign + Ben Thanh is a strong cue for final train boarding.
        if has_platform_sign and has_ben_in_platform:
            return PHASE_AFTER_ESCALATOR_2

        # After ticket gate / platform approach:
        # Ben Thanh + Suoi Tien direction pair means platform-direction decision.
        if pair_ben_suoi is not None:
            return PHASE_AFTER_TICKET

        # Ticket gate / ticket booth area:
        # no-entry + gate pair, ticket sign, gate, or ticket booth.
        if pair_noentry_gate is not None or ocr.has_ticket_sign or ocr.has_gate or ticket_booth is not None:
            return PHASE_AFTER_ESCALATOR_1

        # Entrance / first escalator approach:
        # stair node without ticket/platform cues is treated as entry-side guidance.
        if stair is not None:
            return PHASE_ENTRY

        # If no strong cue, keep previous phase.
        return self.phase

    def _update_scene_phase(self, inferred_phase: int):
        """
        Stabilize phase inference to avoid flickering caused by missed detections.
        """

        if inferred_phase == self.phase:
            self._candidate_phase = inferred_phase
            self._candidate_phase_count = 0
            return

        if inferred_phase == self._candidate_phase:
            self._candidate_phase_count += 1
        else:
            self._candidate_phase = inferred_phase
            self._candidate_phase_count = 1

        if self._candidate_phase_count >= self.PHASE_CONFIRM_FRAMES:
            self.phase = inferred_phase
            self._candidate_phase_count = 0


    # ---------- main update ----------

    def update(self, od: ODResult, ss: SSResult, ocr: OCRResult) -> GuidanceDecision:
        now = time.time()
        decision = GuidanceDecision()
        # == This is new
        # Scene-aware phase recovery.
        # This allows each cut section of the route video to work independently.
        inferred_phase = self._infer_scene_phase(od, ss, ocr)
        self._update_scene_phase(inferred_phase)



        # ticket booth timing (for phase transition)
        for d in od.detections:
            if d.class_name == "ticket booth":
                self.last_ticket_booth_time = now
                self.saw_ticket_booth = True

        # ===================== ESCALATOR GLOBAL HANDLING =====================
        if od.escalator_detections:
            esc = od.escalator_detections[0]

            # stop SS to lighten
            self.ss_enabled = False

            # ALWAYS point arrow to escalator bbox center
            decision.target_pt = esc.center
            decision.clock_dir = esc.clock_dir
            decision.angle_deg = esc.angle_deg

            # Speak+UI: every 3–5s, only if sentence changed
            if esc.clock_dir:
                msg = f"Escalator at {esc.clock_dir}"   # ✅ FIXED (no 'clock' var)
                if self._cooldown_guidance(now, msg):
                    self._commit_guidance(now, msg, force=True)

            decision.ui_text = self.ui_text
            self._escalator_was_visible = True
            return decision  # escalator overrides everything

        # if escalator just disappeared -> phase transition
        # if self._escalator_was_visible:
        #     self._escalator_was_visible = False
        #     if self.phase == PHASE_ENTRY:
        #         self.phase = PHASE_AFTER_ESCALATOR_1
        #     elif self.phase == PHASE_AFTER_TICKET:
        #         self.phase = PHASE_AFTER_ESCALATOR_2
        # == This is new !
        # If escalator just disappeared, do not force a scripted phase jump.
        # The next phase will be inferred from current OCR/OD evidence.
        if self._escalator_was_visible:
            self._escalator_was_visible = False

        # ===================== PHASE LOGIC =====================

        # ---------- PHASE 1: Entry ----------
        if self.phase == PHASE_ENTRY:
            stair = self._od_best(od, "stair node")

            # Global: stair node -> "Tan Cang station" once
            if stair is not None and self._set_once("p1_tancang_once"):
                self._say_and_display("Tan Cang station", force=True)

            # Global guidance to stair node (Move ...):
            # Local override: stand-alone curb ramp when visible -> Follow ...
            if stair is not None:
                if ss.curb_standalone_pt is not None:
                    ok, clock, angle = self._apply_filter(ss.root_xy, ss.curb_standalone_pt)
                    if ok and clock is not None:
                        msg = f"Follow {clock}"
                        if self._cooldown_guidance(now, msg):
                            self._commit_guidance(now, msg, force=True)
                        decision.target_pt = ss.curb_standalone_pt
                        decision.clock_dir = clock
                        decision.angle_deg = angle
                else:
                    # use OD stair target
                    decision.target_pt = stair.center
                    decision.clock_dir = stair.clock_dir
                    decision.angle_deg = stair.angle_deg

                    if stair.clock_dir:
                        msg = f"Move {stair.clock_dir}"
                        if self._cooldown_guidance(now, msg):
                            self._commit_guidance(now, msg)

            decision.ui_text = self.ui_text
            return decision

        # ---------- PHASE 2: After escalator 1 (toward ticket) ----------
        if self.phase == PHASE_AFTER_ESCALATOR_1:
            pair_noentry_gate = self._find_pair(ocr, "no-entry_gate")


            # if noentry_left_gate_right and self._set_once("p2_move_right_ticket_once"):
            #     self.ss_enabled = True
            #     self._say_and_display("Move right heading to ticket booth", force=True)
            #
            #     self.stage2_turnleft_done = False
            #     self.stage2_suppress_ss_until_gate_ahead = False
            #     self.stage2_straight_only_until_ticketbooth = False
            # == This is new: more flexible gate announcement (not only right-gate) !
            if pair_noentry_gate is not None:
                self.ss_enabled = True
                self.stage2_gate_direction_known = True

                if self._set_once("p2_gate_direction_once"):
                    self.stage2_gate_seen_time = now

                    if pair_noentry_gate.left_label == "no-entry" and pair_noentry_gate.right_label == "gate":
                        msg = "Ticket gate is on the right"
                    elif pair_noentry_gate.left_label == "gate" and pair_noentry_gate.right_label == "no-entry":
                        msg = "Ticket gate is on the left"
                    else:
                        msg = "Ticket gate ahead"

                    # Use _commit_guidance, not _say_and_display, so cooldown state is updated.
                    self._commit_guidance(now, msg, force=True)

                    self.stage2_turnleft_done = False
                    self.stage2_suppress_ss_until_gate_ahead = False
                    self.stage2_straight_only_until_ticketbooth = False


            # local: lock to right-edge curb
            # local: lock to right-edge curb
            # == This is new !
            if (not self.stage2_turnleft_done) and (ss.curb_rightedge_pt is not None):
                ok, clock, angle = self._apply_filter(ss.root_xy, ss.curb_rightedge_pt)

                if ok and clock is not None:
                    msg = f"Move {clock}"

                    if self._cooldown_guidance(now, msg):
                        self._commit_guidance(now, msg)

                    decision.target_pt = ss.curb_rightedge_pt
                    decision.clock_dir = clock
                    decision.angle_deg = angle

                    # Do not trigger "Turn left" immediately after detecting gate direction.
                    # This avoids skipping the "Ticket gate is on the right/left" instruction.
                    turnleft_armed = (
                            self.stage2_gate_direction_known
                            and self.stage2_gate_seen_time > 0.0
                            and (now - self.stage2_gate_seen_time) >= self.STAGE2_TURNLEFT_ARM_DELAY_SEC
                    )

                    if turnleft_armed and ss.rightedge_curb_close and self._set_once("p2_turnleft_once"):
                        self.stage2_turnleft_done = True

                        # Do NOT permanently suppress segmentation guidance.
                        # Let fallback / straight guidance continue after the turn-left instruction.
                        self.stage2_suppress_ss_until_gate_ahead = False
                        self.stage2_straight_only_until_ticketbooth = True

                        self._commit_guidance(now, "Turn left", force=True)

                        # Keep the current target for this frame instead of deleting the arrow.
                        # The next frames will continue with SegFormer guidance.
                        decision.target_pt = ss.curb_rightedge_pt
                        decision.clock_dir = clock
                        decision.angle_deg = angle

            # one-time: ticket sign
            if ocr.has_ticket_sign and self._set_once("p2_ticket_counter_once"):
                self._say_and_display("Ticket counter on the right", force=True)

            # one-time: gate ahead when ticket sign AND gate
            if ocr.has_ticket_sign and ocr.has_gate and self._set_once("p2_gate_ahead_once"):
                self._say_and_display("Gate ahead", force=True)

                self.stage2_suppress_ss_until_gate_ahead = False
                self.stage2_straight_only_until_ticketbooth = True

            # if suppressing, do not show any arrow
            if self.stage2_suppress_ss_until_gate_ahead:
                decision.target_pt = None
                decision.clock_dir = None
                decision.angle_deg = 0.0

            # straight-only guidance
            if self.stage2_straight_only_until_ticketbooth and (not self.stage2_suppress_ss_until_gate_ahead):
                candidate = ss.lookahead_pt if ss.mode == "FOLLOWING" else ss.safe_best_pt
                if candidate is not None:
                    ok, clock, angle = self._apply_filter(ss.root_xy, candidate)
                    if ok and clock is not None:
                        verb = "Follow" if ss.mode == "FOLLOWING" else "Move"
                        msg = f"{verb} {clock}"
                        if self._cooldown_guidance(now, msg):
                            self._commit_guidance(now, msg)

                        decision.target_pt = candidate
                        decision.clock_dir = clock
                        decision.angle_deg = angle
                    else:
                        decision.target_pt = None

            # stage2 end -> go to stage3 after ticket booth passed
            if self.saw_ticket_booth and (now - self.last_ticket_booth_time > 3.0):
                if self._set_once("p2_to_p3_once"):
                    self.phase = PHASE_AFTER_TICKET
                    self.stage2_straight_only_until_ticketbooth = False

            # == This is new !!!!
            # Fallback local guidance if no special right-edge curb is available.
            # This prevents stage 2 from becoming silent when the user starts from
            # a different direction or the right-edge curb is not visible.
            if decision.target_pt is None and not self.stage2_suppress_ss_until_gate_ahead:
                candidate = ss.lookahead_pt if ss.mode == "FOLLOWING" else ss.safe_best_pt

                if candidate is not None:
                    ok, clock, angle = self._apply_filter(ss.root_xy, candidate)
                    if ok and clock is not None:
                        verb = "Follow" if ss.mode == "FOLLOWING" else "Move"
                        msg = f"{verb} {clock}"

                        if self._cooldown_guidance(now, msg):
                            self._commit_guidance(now, msg)

                        decision.target_pt = candidate
                        decision.clock_dir = clock
                        decision.angle_deg = angle


            decision.ui_text = self.ui_text
            return decision

        # ---------- PHASE 3: After ticket gate (platform approach, left-only) ----------
        if self.phase == PHASE_AFTER_TICKET:
            pair_ben_suoi = self._find_pair(ocr, "ben-thanh_suoi-tien")
            if pair_ben_suoi is not None:
                side = "left" if pair_ben_suoi.left_label == "ben-thanh-station" else "right"
                if self._set_once("p3_ben_suoi_once") or (now - self.last_ben_thanh_repeat >= 25.0):
                    self.last_ben_thanh_repeat = now
                    self._say_and_display(f"Ben Thanh station is on the {side} turn", force=True)

            stair = self._od_best(od, "stair node")
            if stair is not None and self._set_once("p3_get_on_elevator_once"):
                if stair.clock_dir:
                    self._say_and_display(f"Get on elevator at {stair.clock_dir}", force=True)

            if stair is not None:
                if ss.curb_standalone_pt is not None:
                    ok, clock, angle = self._apply_filter(ss.root_xy, ss.curb_standalone_pt)
                    if ok and clock is not None:
                        msg = f"Move {clock}"
                        if self._cooldown_guidance(now, msg):
                            self._commit_guidance(now, msg)

                        decision.target_pt = ss.curb_standalone_pt
                        decision.clock_dir = clock
                        decision.angle_deg = angle
                else:
                    ok, clock, angle = self._apply_filter(ss.root_xy, stair.center)
                    if ok and clock is not None:
                        msg = f"Move {clock}"
                        if self._cooldown_guidance(now, msg):
                            self._commit_guidance(now, msg)

                        decision.target_pt = stair.center
                        decision.clock_dir = clock
                        decision.angle_deg = angle
                    else:
                        decision.target_pt = None

            else:
                candidate = ss.lookahead_pt if ss.mode == "FOLLOWING" else ss.safe_best_pt
                if candidate is not None:
                    ok, clock, angle = self._apply_filter(ss.root_xy, candidate)
                    if ok and clock is not None:
                        verb = "Follow" if ss.mode == "FOLLOWING" else "Move"
                        msg = f"{verb} {clock}"
                        if self._cooldown_guidance(now, msg):
                            self._commit_guidance(now, msg)

                        decision.target_pt = candidate
                        decision.clock_dir = clock
                        decision.angle_deg = angle

            decision.ui_text = self.ui_text
            return decision

        # ---------- PHASE 4: After escalator 2 (final platform, left-only) ----------
        if self.phase == PHASE_AFTER_ESCALATOR_2:
            if ocr.has_ben_thanh:
                self.ss_enabled = True

            # NEW: faster trigger (no pair needed)
            # If OCR detects a platform-sign and ben-thanh-station appears in any platform,
            # announce immediately (once).
            has_platform_sign = (len(ocr.platforms) > 0)
            has_ben_in_platform = any(
                (c.class_name == "ben-thanh-station")
                for p in ocr.platforms
                for c in p.components
            )

            if has_platform_sign and has_ben_in_platform and self._set_once("p4_turn_left_train_once"):
                self._say_and_display("Turn left to get on train", force=True)

            candidate = ss.lookahead_pt if ss.mode == "FOLLOWING" else ss.safe_best_pt
            if ss.curb_leftturn_pt is not None:
                candidate = ss.curb_leftturn_pt

            if candidate is not None:
                ok, clock, angle = self._apply_filter(ss.root_xy, candidate)
                if ok and clock is not None:
                    verb = "Follow" if ss.mode == "FOLLOWING" else "Move"
                    msg = f"{verb} {clock}"
                    if self._cooldown_guidance(now, msg):
                        self._commit_guidance(now, msg)

                    decision.target_pt = candidate
                    decision.clock_dir = clock
                    decision.angle_deg = angle

            # gate states (OD) - one time speech, UI persists until next spoken message
            for d in od.detections:
                if d.class_name == "closed gate" and d.clock_dir == "12 o'clock" and self._set_once("p4_closed_gate_once"):
                    self._say_and_display("Door is closing, wait for the train to come!", force=True)
                if d.class_name == "open gate" and d.clock_dir == "12 o'clock" and self._set_once("p4_open_gate_once"):
                    self._say_and_display("The door is open, please get on the train!", force=True)

            decision.ui_text = self.ui_text
            return decision

        # fallback
        decision.ui_text = self.ui_text
        return decision
