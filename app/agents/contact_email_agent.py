import uuid
import smtplib
import logging
import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session

from app.config import settings
from app.models import ContactInquiry

logger = logging.getLogger("ContactEmailAgent")

TARGET_RECIPIENT_EMAIL = getattr(settings, "SUPPORT_EMAIL", "battulasarath751@gmail.com")

class ContactEmailAgent:
    """
    AGENT 4: CONTACT US & EMAIL AGENT
    Specialized in processing healthcare governance and clinical support inquiries from users,
    generating unique tracking ticket IDs, storing audit records, and sending automated emails
    directly to battulasarath751@gmail.com.
    """
    def __init__(self, db: Session, target_email: Optional[str] = None):
        self.db = db
        self.target_email = target_email or TARGET_RECIPIENT_EMAIL

    def process_and_send_email(
        self,
        name: str,
        user_email: str,
        category: str,
        subject: str,
        message: str
    ) -> Dict[str, Any]:
        ticket_id = f"HLTH-{uuid.uuid4().hex[:8].upper()}"
        timestamp_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        # 1. Format Email Body
        plain_text = (
            f"=== NEW HEALTHCARE INQUIRY RECEIVED ===\n"
            f"Ticket ID: {ticket_id}\n"
            f"Date/Time: {timestamp_str}\n"
            f"Sender Name: {name}\n"
            f"Sender Email: {user_email}\n"
            f"Inquiry Category: {category}\n"
            f"Subject: {subject}\n\n"
            f"Message / Question:\n{message}\n\n"
            f"======================================\n"
            f"MediCare AI Clinical Governance Agent\n"
        )

        html_text = f"""
        <html>
        <body style="font-family: Arial, sans-serif; background-color: #0f172a; color: #f8fafc; padding: 20px;">
            <div style="max-width: 600px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 25px; border: 1px solid #06b6d4;">
                <h2 style="color: #06b6d4; margin-top: 0;">🏥 MediCare AI - New Support Inquiry</h2>
                <div style="background: #0f172a; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 4px solid #10b981;">
                    <p style="margin: 5px 0;"><strong>Ticket ID:</strong> <span style="color: #10b981;">{ticket_id}</span></p>
                    <p style="margin: 5px 0;"><strong>Submitted At:</strong> {timestamp_str}</p>
                    <p style="margin: 5px 0;"><strong>Sender Name:</strong> {name}</p>
                    <p style="margin: 5px 0;"><strong>Sender Email:</strong> <a href="mailto:{user_email}" style="color: #38bdf8;">{user_email}</a></p>
                    <p style="margin: 5px 0;"><strong>Category:</strong> {category}</p>
                </div>
                <h4 style="color: #e2e8f0; margin-bottom: 8px;">Subject: {subject}</h4>
                <div style="background: #334155; padding: 15px; border-radius: 8px; font-size: 15px; line-height: 1.6; white-space: pre-wrap;">
{message}
                </div>
                <hr style="border: 0; border-top: 1px solid #475569; margin: 25px 0;" />
                <p style="font-size: 12px; color: #94a3b8; text-align: center;">
                    This email was automatically dispatched by <strong>ContactEmailAgent</strong> to {self.target_email}.
                </p>
            </div>
        </body>
        </html>
        """

        # 2. Attempt HTTPS Web Email Dispatch (Port 443 - Bypasses ISP SMTP port blocking)
        email_status = "SENT_VIA_HTTPS"
        https_success = False

        try:
            import urllib.request
            import urllib.parse
            
            form_payload = urllib.parse.urlencode({
                "name": name,
                "email": user_email,
                "subject": subject,
                "message": message,
                "category": category,
                "ticket_id": ticket_id,
                "_subject": f"[MediCare AI - {ticket_id}] {subject}",
                "_captcha": "false"
            }).encode('utf-8')

            req = urllib.request.Request(
                f"https://formsubmit.co/ajax/{self.target_email}",
                data=form_payload,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Referer": "http://127.0.0.1:5000/"
                }
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                resp_json = resp.read().decode('utf-8')
                logger.info(f"HTTPS Dispatch for ticket {ticket_id} to {self.target_email}: {resp_json}")
                email_status = "DISPATCHED_TO_GMAIL_INBOX"
                https_success = True
        except Exception as https_err:
            logger.warning(f"HTTPS web dispatch fallback: {https_err}")

        # 3. Secondary Attempt: Standard SMTP Dispatch if configured
        smtp_server = getattr(settings, "SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(getattr(settings, "SMTP_PORT", 587))
        smtp_user = getattr(settings, "SMTP_USERNAME", "")
        smtp_pass = getattr(settings, "SMTP_PASSWORD", "")
        from_email = getattr(settings, "SMTP_FROM_EMAIL", "noreply.medicare.ai@gmail.com")

        if smtp_user and smtp_pass and not https_success:
            try:
                msg = MIMEMultipart("alternative")
                msg["Subject"] = f"[MediCare AI Support - {ticket_id}] {subject}"
                msg["From"] = from_email or smtp_user
                msg["To"] = self.target_email
                msg["Reply-To"] = user_email

                msg.attach(MIMEText(plain_text, "plain"))
                msg.attach(MIMEText(html_text, "html"))

                with smtplib.SMTP(smtp_server, smtp_port, timeout=4) as server:
                    server.starttls()
                    server.login(smtp_user, smtp_pass)
                    server.sendmail(from_email or smtp_user, [self.target_email], msg.as_string())
                
                logger.info(f"Successfully sent email for ticket {ticket_id} to {self.target_email}")
                email_status = "SENT_VIA_SMTP"
            except Exception as e:
                logger.warning(f"SMTP dispatch failed ({str(e)}). Storing inquiry in database.")
                if not https_success:
                    email_status = f"STORED_IN_DB ({str(e)[:30]})"

        # 3. Store in Database
        try:
            inquiry_record = ContactInquiry(
                ticket_id=ticket_id,
                name=name,
                email=user_email,
                category=category,
                subject=subject,
                message=message,
                recipient_email=self.target_email,
                email_status=email_status
            )
            self.db.add(inquiry_record)
            self.db.commit()
        except Exception as db_err:
            self.db.rollback()
            logger.error(f"Failed to save inquiry to DB: {db_err}")

        return {
            "ticket_id": ticket_id,
            "status": "RECEIVED",
            "email_dispatched_to": self.target_email,
            "email_status": email_status,
            "message": f"Thank you, {name}. Your inquiry has been processed by our Contact & Email Agent and dispatched to {self.target_email}.",
            "emergency_notice": "If this is an acute emergency, call 112 or 108 immediately.",
            "received_at": timestamp_str
        }
