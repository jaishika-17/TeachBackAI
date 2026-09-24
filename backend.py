"""
backend.py -- TeachBack AI's persistence + API layer.

Run with:
    pip install fastapi "uvicorn[standard]" sqlalchemy pydantic
    uvicorn backend:app --reload --port 8000

This file owns the database, accounts, and the HTTP surface. It knows
nothing about *how* an explanation gets evaluated -- every bit of that
intelligence (the knowledge base, scoring, gap detection, adaptive
questions, repair content, verification thresholds) lives in ai.py and is
called through the small set of functions imported below.

Auth note: passwords are hashed with PBKDF2-SHA256 (stdlib `hashlib`, no
extra dependency) and salted per-user. There's no session token / JWT --
the frontend just remembers the student id returned by login. That's
fine for a project like this; swap in a real token scheme before putting
any real user's data behind it.

SQLite is used by default (zero setup); a single file `teachback.db` is
created next to this script on first run, which also seeds the Concept
table from ai.CONCEPTS.
"""
import hashlib
import os
import secrets
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON, Boolean
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session

import ai

# ---------------------------------------------------------------------------
# Database setup
# ---------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./teachback.db")
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Password hashing (stdlib only)
# ---------------------------------------------------------------------------
def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()
    return digest, salt


def verify_password(password: str, salt: str, digest: str) -> bool:
    check, _ = hash_password(password, salt)
    return secrets.compare_digest(check, digest)


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------
class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=True)
    password_hash = Column(String, nullable=True)
    password_salt = Column(String, nullable=True)
    student_code = Column(String, nullable=True)   # the learner's own ID/roll number, informational only
    name = Column(String, default="Learner")
    education_level = Column(String, nullable=True)
    preferred_language = Column(String, default="en")
    created_at = Column(DateTime, default=datetime.utcnow)


class Concept(Base):
    """
    Lightweight metadata row only -- the actual knowledge (key points,
    misconceptions) lives in ai.CONCEPTS, keyed by `slug`. This table
    exists so sessions/profiles can hold a stable foreign key and the
    library can be listed/paginated like normal data.
    """
    __tablename__ = "concepts"
    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String, unique=True, index=True)
    name = Column(String)
    subject = Column(String)
    difficulty = Column(String)
    description = Column(Text)


class TeachBackSession(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    concept_id = Column(Integer, ForeignKey("concepts.id"))
    self_confidence = Column(Integer, nullable=True)
    status = Column(String, default="IN_PROGRESS")
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    concept = relationship("Concept")
    attempts = relationship("ExplanationAttempt", back_populates="session", cascade="all, delete-orphan")
    quiz_answers = relationship("QuizAnswer", back_populates="session", cascade="all, delete-orphan", order_by="QuizAnswer.id")


class ExplanationAttempt(Base):
    __tablename__ = "explanation_attempts"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"))
    attempt_number = Column(Integer, default=1)
    explanation_text = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("TeachBackSession", back_populates="attempts")
    evaluation = relationship("Evaluation", back_populates="attempt", uselist=False, cascade="all, delete-orphan")


class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True, index=True)
    attempt_id = Column(Integer, ForeignKey("explanation_attempts.id"))
    understanding_score = Column(Float)
    accuracy_score = Column(Float)
    reasoning_score = Column(Float)
    completeness_score = Column(Float)
    application_score = Column(Float)
    status = Column(String)

    attempt = relationship("ExplanationAttempt", back_populates="evaluation")
    gaps = relationship("DetectedGap", back_populates="evaluation", cascade="all, delete-orphan")
    challenges = relationship("Challenge", back_populates="evaluation", cascade="all, delete-orphan")


class DetectedGap(Base):
    __tablename__ = "detected_gaps"
    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"))
    gap_type = Column(String)
    target_label = Column(String, nullable=True)
    is_misconception = Column(Boolean, default=False)
    misconception_index = Column(Integer, nullable=True)
    severity = Column(Integer, default=2)
    description = Column(Text)

    evaluation = relationship("Evaluation", back_populates="gaps")


class Challenge(Base):
    __tablename__ = "challenges"
    id = Column(Integer, primary_key=True, index=True)
    evaluation_id = Column(Integer, ForeignKey("evaluations.id"))
    question = Column(Text)
    target_label = Column(String, nullable=True)
    is_misconception = Column(Boolean, default=False)
    misconception_index = Column(Integer, nullable=True)
    difficulty = Column(String, default="medium")
    level = Column(String, default="basic")   # "basic" (confidence 1-3) | "advanced" (confidence 4-5)
    student_answer = Column(Text, nullable=True)
    result = Column(String, default="PENDING")

    evaluation = relationship("Evaluation", back_populates="challenges")


class QuizAnswer(Base):
    """One answered multiple-choice question inside a session (quiz flow)."""
    __tablename__ = "quiz_answers"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), index=True)
    question_id = Column(String, index=True)
    selected_index = Column(Integer)
    is_correct = Column(Boolean, default=False)
    answered_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("TeachBackSession", back_populates="quiz_answers")


class KnowledgeProfile(Base):
    __tablename__ = "knowledge_profiles"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    concept_id = Column(Integer, ForeignKey("concepts.id"))
    understanding_score = Column(Float, default=0)
    accuracy_score = Column(Float, default=0)
    reasoning_score = Column(Float, default=0)
    completeness_score = Column(Float, default=0)
    verification_status = Column(String, default="not_started")
    attempts_count = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow)

    concept = relationship("Concept")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class RegisterIn(BaseModel):
    username: str = Field(..., min_length=3, max_length=40)
    password: str = Field(..., min_length=4)
    name: str = "Learner"
    student_code: Optional[str] = None


class LoginIn(BaseModel):
    username: str
    password: str


class StudentOut(BaseModel):
    id: int
    username: Optional[str]
    name: str
    student_code: Optional[str]
    education_level: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class ConceptOut(BaseModel):
    id: int
    slug: str
    name: str
    subject: str
    difficulty: str
    description: str

    class Config:
        from_attributes = True


class SessionCreate(BaseModel):
    student_id: int
    concept_id: int
    self_confidence: Optional[int] = Field(None, ge=1, le=5)


class SessionOut(BaseModel):
    id: int
    student_id: int
    concept_id: int
    status: str

    class Config:
        from_attributes = True


class ExplanationIn(BaseModel):
    text: str = Field(..., min_length=5)


class GapOut(BaseModel):
    id: int
    gap_type: str
    target_label: Optional[str]
    severity: int
    description: str

    class Config:
        from_attributes = True


class ChallengeOut(BaseModel):
    id: int
    question: str
    target_label: Optional[str]
    difficulty: str
    level: str
    result: str

    class Config:
        from_attributes = True


class EvaluationOut(BaseModel):
    id: int
    understanding_score: float
    accuracy_score: float
    reasoning_score: float
    completeness_score: float
    application_score: float
    status: str
    gaps: list[GapOut]
    challenge: Optional[ChallengeOut] = None
    session_status: str


class ChallengeAnswerIn(BaseModel):
    text: str = Field(..., min_length=1)


class RepairOut(BaseModel):
    repair_type: str
    content: dict


class EvalSummary(BaseModel):
    understanding_score: float
    accuracy_score: float
    reasoning_score: float
    completeness_score: float
    status: str


class ChallengeAnswerOut(BaseModel):
    """
    The response that drives the final "Verification" screen: the graded
    challenge, a synthesized answer combining what the student said with
    the authoritative explanation, the always-available deeper-detail
    content, the updated (cumulative) evaluation, and the verdict.
    """
    challenge: ChallengeOut
    combined_answer: str
    detail: RepairOut
    evaluation: EvalSummary
    verdict: str
    session_status: str


class QuestionOut(BaseModel):
    id: str
    question: str
    options: list[str]
    level: str
    number: int      # 1-based position of this question in the session
    total: int       # how many questions exist for this concept


class ProgressOut(BaseModel):
    answered: int
    correct: int
    percent: int


class NextQuestionOut(BaseModel):
    done: bool                       # True when every question has been answered
    question: Optional[QuestionOut] = None
    progress: ProgressOut


class QuizAnswerIn(BaseModel):
    question_id: str
    selected_index: int = Field(..., ge=0)


class QuizAnswerOut(BaseModel):
    is_correct: bool
    selected_index: int
    correct_index: int
    explanation: str
    progress: ProgressOut
    has_more: bool                   # are there still unanswered questions?


class ReviewItemOut(BaseModel):
    question_id: str
    question: str
    options: list[str]
    selected_index: int
    correct_index: int
    is_correct: bool
    explanation: str


class QuizSummaryOut(BaseModel):
    session_id: int
    concept_name: str
    answered: int
    correct: int
    percent: int
    verdict: str                     # VERIFIED | NEEDS_REINFORCEMENT
    review: list[ReviewItemOut]


class ProfileOut(BaseModel):
    concept_id: int
    concept_name: str
    understanding_score: float
    accuracy_score: float
    reasoning_score: float
    completeness_score: float
    verification_status: str
    attempts_count: int


class WeakAreaOut(BaseModel):
    concept_name: str
    gap_type: str
    description: str


class HistoryItemOut(BaseModel):
    session_id: int
    concept_name: str
    started_at: datetime
    status: str
    attempts_count: int
    final_understanding: float


# ---------------------------------------------------------------------------
# App + startup seeding
# ---------------------------------------------------------------------------
app = FastAPI(title="TeachBack AI API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if db.query(Concept).count() == 0:
            for slug, meta in ai.list_concepts_meta_by_slug().items():
                db.add(Concept(slug=slug, **meta))
            db.commit()
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.post("/auth/register", response_model=StudentOut)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    if db.query(Student).filter_by(username=payload.username).first():
        raise HTTPException(400, "That username is already taken")
    digest, salt = hash_password(payload.password)
    student = Student(
        username=payload.username, password_hash=digest, password_salt=salt,
        name=payload.name, student_code=payload.student_code,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


@app.post("/auth/login", response_model=StudentOut)
def login(payload: LoginIn, db: Session = Depends(get_db)):
    student = db.query(Student).filter_by(username=payload.username).first()
    if not student or not student.password_hash or not verify_password(payload.password, student.password_salt, student.password_hash):
        raise HTTPException(401, "Incorrect username or password")
    return student


@app.get("/students/{student_id}", response_model=StudentOut)
def get_student(student_id: int, db: Session = Depends(get_db)):
    student = db.get(Student, student_id)
    if not student:
        raise HTTPException(404, "Student not found")
    return student


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------
@app.get("/concepts", response_model=list[ConceptOut])
def list_concepts(db: Session = Depends(get_db)):
    return db.query(Concept).order_by(Concept.id).all()


@app.get("/concepts/{concept_id}")
def get_concept(concept_id: int, db: Session = Depends(get_db)):
    concept = db.get(Concept, concept_id)
    if not concept:
        raise HTTPException(404, "Concept not found")
    detail = ai.get_concept_detail(concept.slug)
    return {"id": concept.id, **detail}


# ---------------------------------------------------------------------------
# Sessions + the core TeachBack loop
# ---------------------------------------------------------------------------
def _upsert_profile(db: Session, session: TeachBackSession, result: dict, verdict: str) -> KnowledgeProfile:
    profile = db.query(KnowledgeProfile).filter_by(
        student_id=session.student_id, concept_id=session.concept_id
    ).first()
    if not profile:
        profile = KnowledgeProfile(student_id=session.student_id, concept_id=session.concept_id)
        db.add(profile)

    profile.understanding_score = result["understanding"]
    profile.accuracy_score = result["accuracy"]
    profile.reasoning_score = result["reasoning"]
    profile.completeness_score = result["completeness"]
    profile.verification_status = verdict
    profile.attempts_count = (profile.attempts_count or 0) + 1
    profile.last_updated = datetime.utcnow()
    return profile


def _record_and_evaluate(db: Session, session: TeachBackSession, slug: str, text: str):
    """
    Records `text` as a new attempt (whether it came from the main
    explanation box or from answering a challenge) and evaluates it
    against the FULL cumulative text of the session so far. This is the
    fix for scores that could never climb high enough to verify: without
    it, re-explaining just the missing piece made previously-covered
    points "disappear" because they weren't retyped in the new attempt.
    """
    prior_text = " ".join(a.explanation_text for a in session.attempts)
    cumulative = (prior_text + " " + text).strip()

    attempt = ExplanationAttempt(
        session_id=session.id, attempt_number=len(session.attempts) + 1, explanation_text=text,
    )
    db.add(attempt)
    db.flush()

    result = ai.evaluate_explanation(slug, cumulative)

    evaluation = Evaluation(
        attempt_id=attempt.id,
        understanding_score=result["understanding"], accuracy_score=result["accuracy"],
        reasoning_score=result["reasoning"], completeness_score=result["completeness"],
        application_score=result["application"], status=result["status"],
    )
    db.add(evaluation)
    db.flush()

    gap_rows = []
    for g in result["gaps"]:
        row = DetectedGap(
            evaluation_id=evaluation.id, gap_type=g["gap_type"], target_label=g["target_label"],
            is_misconception=g["is_misconception"], misconception_index=g["misconception_index"],
            severity=g["severity"], description=g["description"],
        )
        db.add(row)
        gap_rows.append(row)
    db.flush()

    return attempt, evaluation, gap_rows, result


def _gap_still_open(gaps: list, target_label, is_misconception: bool, misconception_index) -> bool:
    return any(
        g["target_label"] == target_label and g["is_misconception"] == is_misconception
        and g["misconception_index"] == misconception_index
        for g in gaps
    )


def _finalize(db: Session, session: TeachBackSession, result: dict, resolved_target_gap: bool) -> str:
    verdict = ai.check_verification(result, resolved_target_gap)
    _upsert_profile(db, session, result, verdict)
    if verdict == "VERIFIED":
        session.status = "VERIFIED"
        session.completed_at = datetime.utcnow()
    else:
        session.status = "IN_PROGRESS"
    return verdict


@app.post("/sessions", response_model=SessionOut)
def start_session(payload: SessionCreate, db: Session = Depends(get_db)):
    if not db.get(Student, payload.student_id) or not db.get(Concept, payload.concept_id):
        raise HTTPException(404, "Student or concept not found")
    session = TeachBackSession(**payload.model_dump())
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@app.get("/sessions/{session_id}", response_model=SessionOut)
def get_session(session_id: int, db: Session = Depends(get_db)):
    session = db.get(TeachBackSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@app.post("/sessions/{session_id}/explanation", response_model=EvaluationOut)
def submit_explanation(session_id: int, payload: ExplanationIn, db: Session = Depends(get_db)):
    session = db.get(TeachBackSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    slug = session.concept.slug

    attempt, evaluation, gap_rows, result = _record_and_evaluate(db, session, slug, payload.text)

    # was the gap that triggered the *previous* challenge resolved this time?
    resolved_target_gap = True
    prior_challenge = (
        db.query(Challenge).join(Evaluation).join(ExplanationAttempt)
        .filter(ExplanationAttempt.session_id == session.id)
        .order_by(Challenge.id.desc()).first()
    )
    if prior_challenge:
        resolved_target_gap = not _gap_still_open(
            result["gaps"], prior_challenge.target_label, prior_challenge.is_misconception, prior_challenge.misconception_index
        )

    verdict = _finalize(db, session, result, resolved_target_gap)

    challenge_out = None
    if verdict != "VERIFIED" and gap_rows:
        top = result["gaps"][0]
        level = ai.level_for_confidence(session.self_confidence)
        challenge_data = ai.generate_challenge(slug, top, level)
        challenge = Challenge(
            evaluation_id=evaluation.id, question=challenge_data["question"],
            target_label=challenge_data["target_label"], is_misconception=challenge_data["is_misconception"],
            misconception_index=challenge_data["misconception_index"], difficulty=challenge_data["difficulty"],
            level=challenge_data["level"],
        )
        db.add(challenge)
        db.flush()
        challenge_out = challenge

    db.commit()
    db.refresh(evaluation)

    return EvaluationOut(
        id=evaluation.id, understanding_score=evaluation.understanding_score,
        accuracy_score=evaluation.accuracy_score, reasoning_score=evaluation.reasoning_score,
        completeness_score=evaluation.completeness_score, application_score=evaluation.application_score,
        status=evaluation.status, gaps=gap_rows, challenge=challenge_out, session_status=session.status,
    )


@app.get("/challenges/{challenge_id}", response_model=ChallengeOut)
def get_challenge(challenge_id: int, db: Session = Depends(get_db)):
    challenge = db.get(Challenge, challenge_id)
    if not challenge:
        raise HTTPException(404, "Challenge not found")
    return challenge


@app.post("/challenges/{challenge_id}/answer", response_model=ChallengeAnswerOut)
def answer_challenge(challenge_id: int, payload: ChallengeAnswerIn, db: Session = Depends(get_db)):
    """
    Grades the answer, then folds it straight into the session's
    cumulative evaluation and returns everything the final Verification
    screen needs in one round trip: the verdict, a synthesized answer
    (the student's own words + the authoritative explanation), and the
    always-available "more detail" content -- no separate repair step,
    no forced loop back through re-explaining to reach this point.
    """
    challenge = db.get(Challenge, challenge_id)
    if not challenge:
        raise HTTPException(404, "Challenge not found")

    session = challenge.evaluation.attempt.session
    slug = session.concept.slug

    grade = ai.grade_challenge_answer(slug, challenge.target_label, challenge.is_misconception, payload.text)
    challenge.student_answer = payload.text
    challenge.result = grade
    db.flush()

    attempt, evaluation, gap_rows, result = _record_and_evaluate(db, session, slug, payload.text)
    resolved = not _gap_still_open(result["gaps"], challenge.target_label, challenge.is_misconception, challenge.misconception_index)
    verdict = _finalize(db, session, result, resolved)

    combined_answer = ai.compose_combined_answer(
        slug, challenge.target_label, challenge.is_misconception, challenge.misconception_index, grade, payload.text,
    )
    detail_data = ai.generate_repair(slug, challenge.target_label, challenge.is_misconception, challenge.misconception_index)

    db.commit()

    return ChallengeAnswerOut(
        challenge=challenge,
        combined_answer=combined_answer,
        detail=RepairOut(repair_type=detail_data["repair_type"], content=detail_data["content"]),
        evaluation=EvalSummary(
            understanding_score=result["understanding"], accuracy_score=result["accuracy"],
            reasoning_score=result["reasoning"], completeness_score=result["completeness"], status=result["status"],
        ),
        verdict=verdict,
        session_status=session.status,
    )


# ---------------------------------------------------------------------------
# Quiz flow:  confidence -> question -> check -> performance -> more? -> final page
# ---------------------------------------------------------------------------
def _quiz_pool(session: TeachBackSession) -> list:
    """Ordered question list for this session (level chosen from the confidence rating)."""
    level = ai.level_for_confidence(session.self_confidence)
    return ai.get_question_pool(session.concept.slug, level)


def _quiz_progress(session: TeachBackSession) -> ProgressOut:
    answered = len(session.quiz_answers)
    correct = sum(1 for a in session.quiz_answers if a.is_correct)
    percent = round(100 * correct / answered) if answered else 0
    return ProgressOut(answered=answered, correct=correct, percent=percent)


def _get_session_or_404(db: Session, session_id: int) -> TeachBackSession:
    session = db.get(TeachBackSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return session


@app.get("/sessions/{session_id}/next-question", response_model=NextQuestionOut)
def next_question(session_id: int, db: Session = Depends(get_db)):
    """The next unanswered question. The correct answer is never sent to the browser here."""
    session = _get_session_or_404(db, session_id)
    pool = _quiz_pool(session)
    answered_ids = {a.question_id for a in session.quiz_answers}
    progress = _quiz_progress(session)

    for position, q in enumerate(pool, start=1):
        if q["id"] not in answered_ids:
            return NextQuestionOut(
                done=False, progress=progress,
                question=QuestionOut(
                    id=q["id"], question=q["question"], options=q["options"], level=q["level"],
                    number=len(answered_ids) + 1, total=len(pool),
                ),
            )
    return NextQuestionOut(done=True, question=None, progress=progress)


@app.post("/sessions/{session_id}/answer", response_model=QuizAnswerOut)
def answer_question(session_id: int, payload: QuizAnswerIn, db: Session = Depends(get_db)):
    """Checks one answer, stores it, and returns the result plus the running performance."""
    session = _get_session_or_404(db, session_id)
    if session.completed_at is not None:
        raise HTTPException(400, "This session is already finished")

    pool = _quiz_pool(session)
    q = next((x for x in pool if x["id"] == payload.question_id), None)
    if q is None:
        raise HTTPException(404, "Question not found for this session")
    if payload.selected_index >= len(q["options"]):
        raise HTTPException(400, "Invalid option selected")

    existing = next((a for a in session.quiz_answers if a.question_id == q["id"]), None)
    if existing is None:   # a double click / retry must not count twice
        existing = QuizAnswer(
            session_id=session.id, question_id=q["id"], selected_index=payload.selected_index,
            is_correct=(payload.selected_index == q["answer"]),
        )
        db.add(existing)
        db.commit()
        db.refresh(session)

    answered_ids = {a.question_id for a in session.quiz_answers}
    return QuizAnswerOut(
        is_correct=existing.is_correct, selected_index=existing.selected_index, correct_index=q["answer"],
        explanation=q["explanation"], progress=_quiz_progress(session),
        has_more=any(x["id"] not in answered_ids for x in pool),
    )


def _quiz_summary(session: TeachBackSession) -> QuizSummaryOut:
    slug = session.concept.slug
    review = []
    for a in session.quiz_answers:
        q = ai.get_question(slug, a.question_id)
        if not q:
            continue
        review.append(ReviewItemOut(
            question_id=q["id"], question=q["question"], options=q["options"],
            selected_index=a.selected_index, correct_index=q["answer"],
            is_correct=a.is_correct, explanation=q["explanation"],
        ))
    p = _quiz_progress(session)
    return QuizSummaryOut(
        session_id=session.id, concept_name=session.concept.name, answered=p.answered,
        correct=p.correct, percent=p.percent, verdict=ai.quiz_verdict(p.percent), review=review,
    )


@app.post("/sessions/{session_id}/finish", response_model=QuizSummaryOut)
def finish_session(session_id: int, db: Session = Depends(get_db)):
    """Ends the quiz: saves the result to the knowledge profile and returns the final report."""
    session = _get_session_or_404(db, session_id)
    if not session.quiz_answers:
        raise HTTPException(400, "Answer at least one question before finishing")

    summary = _quiz_summary(session)
    if session.completed_at is None:   # idempotent: a page refresh on the final screen is safe
        profile = db.query(KnowledgeProfile).filter_by(
            student_id=session.student_id, concept_id=session.concept_id
        ).first()
        if not profile:
            profile = KnowledgeProfile(student_id=session.student_id, concept_id=session.concept_id)
            db.add(profile)
        # The quiz produces one number (% correct); it feeds every score column the dashboard reads.
        profile.understanding_score = profile.accuracy_score = summary.percent
        profile.reasoning_score = profile.completeness_score = summary.percent
        profile.verification_status = summary.verdict
        profile.attempts_count = (profile.attempts_count or 0) + 1
        profile.last_updated = datetime.utcnow()

        session.status = summary.verdict
        session.completed_at = datetime.utcnow()
        db.commit()
    return summary


@app.get("/sessions/{session_id}/summary", response_model=QuizSummaryOut)
def session_summary(session_id: int, db: Session = Depends(get_db)):
    return _quiz_summary(_get_session_or_404(db, session_id))


# ---------------------------------------------------------------------------
# Progress / history
# ---------------------------------------------------------------------------
@app.get("/students/{student_id}/progress", response_model=list[ProfileOut])
def get_progress(student_id: int, db: Session = Depends(get_db)):
    profiles = db.query(KnowledgeProfile).filter_by(student_id=student_id).all()
    return [
        ProfileOut(
            concept_id=p.concept_id, concept_name=p.concept.name, understanding_score=p.understanding_score,
            accuracy_score=p.accuracy_score, reasoning_score=p.reasoning_score,
            completeness_score=p.completeness_score, verification_status=p.verification_status,
            attempts_count=p.attempts_count,
        )
        for p in profiles
    ]


@app.get("/students/{student_id}/weak-areas", response_model=list[WeakAreaOut])
def get_weak_areas(student_id: int, db: Session = Depends(get_db)):
    sessions = db.query(TeachBackSession).filter_by(student_id=student_id).order_by(TeachBackSession.id).all()
    out = []

    # Quiz sessions: for each concept, list what was answered wrong in the most recent quiz.
    latest_quiz = {}
    for session in sessions:
        if session.quiz_answers:
            latest_quiz[session.concept_id] = session
    for session in latest_quiz.values():
        for a in session.quiz_answers:
            if a.is_correct:
                continue
            q = ai.get_question(session.concept.slug, a.question_id)
            if q:
                out.append(WeakAreaOut(
                    concept_name=session.concept.name, gap_type="INCORRECT_ANSWER",
                    description=f"{q['question']}  Correct answer: {q['options'][q['answer']]}",
                ))

    for session in sessions:
        if session.quiz_answers or not session.attempts:
            continue
        latest = max(session.attempts, key=lambda a: a.attempt_number)
        if not latest.evaluation:
            continue
        for gap in latest.evaluation.gaps:
            out.append(WeakAreaOut(concept_name=session.concept.name, gap_type=gap.gap_type, description=gap.description))
    return out


@app.get("/students/{student_id}/history", response_model=list[HistoryItemOut])
def get_history(student_id: int, db: Session = Depends(get_db)):
    sessions = (
        db.query(TeachBackSession).filter_by(student_id=student_id)
        .order_by(TeachBackSession.started_at.desc()).all()
    )
    out = []
    for s in sessions:
        if s.quiz_answers:
            p = _quiz_progress(s)
            out.append(HistoryItemOut(
                session_id=s.id, concept_name=s.concept.name, started_at=s.started_at,
                status=s.status, attempts_count=p.answered, final_understanding=p.percent,
            ))
            continue
        if not s.attempts:
            continue
        latest = max(s.attempts, key=lambda a: a.attempt_number)
        understanding = latest.evaluation.understanding_score if latest.evaluation else 0.0
        out.append(HistoryItemOut(
            session_id=s.id, concept_name=s.concept.name, started_at=s.started_at,
            status=s.status, attempts_count=len(s.attempts), final_understanding=understanding,
        ))
    return out