"""
Email Service for Financial Analyzer
Supports SendGrid and fallback to console logging for development
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Try to import SendGrid
try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Email, To, Content
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmailService:
    """Email service with SendGrid integration and fallback support"""
    
    def __init__(self):
        self.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        self.from_email = os.getenv("FROM_EMAIL", "noreply@moneta-financial.com")
        self.from_name = os.getenv("FROM_NAME", "MONETA Financial Analyzer")
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # Initialize SendGrid client if API key is available
        self.sendgrid_client = None
        if self.sendgrid_api_key and SENDGRID_AVAILABLE:
            try:
                self.sendgrid_client = SendGridAPIClient(api_key=self.sendgrid_api_key)
                logger.info("SendGrid email service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize SendGrid: {e}")
                self.sendgrid_client = None
        elif not SENDGRID_AVAILABLE:
            logger.warning("SendGrid library not installed. Email service will use console logging only.")
        elif not self.sendgrid_api_key:
            logger.info("SENDGRID_API_KEY not set. Email service will use console logging only.")
    
    def _send_email_sendgrid(self, to_email: str, subject: str, html_content: str, text_content: Optional[str] = None) -> bool:
        """Send email using SendGrid"""
        if not self.sendgrid_client:
            return False
        
        try:
            message = Mail(
                from_email=Email(self.from_email, self.from_name),
                to_emails=To(to_email),
                subject=subject,
                html_content=Content("text/html", html_content)
            )
            
            # Add plain text version if provided
            if text_content:
                message.plain_text_content = Content("text/plain", text_content)
            
            response = self.sendgrid_client.send(message)
            
            # Check if email was sent successfully (status code 202)
            if response.status_code == 202:
                logger.info(f"Email sent successfully to {to_email}")
                return True
            else:
                logger.error(f"Failed to send email to {to_email}. Status code: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending email via SendGrid to {to_email}: {str(e)}")
            return False
    
    def _send_email_fallback(self, to_email: str, subject: str, html_content: str, text_content: Optional[str] = None) -> bool:
        """Fallback email logging for development/testing"""
        logger.info(f"[EMAIL SERVICE] Would send email to: {to_email}")
        logger.info(f"[EMAIL SERVICE] Subject: {subject}")
        if text_content:
            logger.info(f"[EMAIL SERVICE] Content:\n{text_content}")
        else:
            # Extract text from HTML for logging
            import re
            text = re.sub(r'<[^>]+>', '', html_content)
            logger.info(f"[EMAIL SERVICE] Content:\n{text}")
        return True
    
    def send_email(self, to_email: str, subject: str, html_content: str, text_content: Optional[str] = None) -> bool:
        """
        Send email using SendGrid if available, otherwise fallback to console logging
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email content
            text_content: Optional plain text version
            
        Returns:
            bool: True if email was sent successfully (or logged in dev mode), False otherwise
        """
        # Try SendGrid first if available
        if self.sendgrid_client and self.environment != "development":
            success = self._send_email_sendgrid(to_email, subject, html_content, text_content)
            if success:
                return True
        
        # Fallback to console logging (development mode or SendGrid unavailable)
        return self._send_email_fallback(to_email, subject, html_content, text_content)
    
    def send_password_reset_email(self, to_email: str, reset_token: str, username: Optional[str] = None) -> bool:
        """
        Send password reset email
        
        Args:
            to_email: Recipient email address
            reset_token: Password reset token
            username: Optional username for personalization
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        reset_link = f"https://moneta-backend-api.onrender.com/api/auth/reset-password?token={reset_token}"
        
        # HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #2563eb; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
                .content {{ background-color: #f9fafb; padding: 30px; border-radius: 0 0 5px 5px; }}
                .button {{ display: inline-block; padding: 12px 30px; background-color: #2563eb; color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
                .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
                .warning {{ background-color: #fef3c7; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #f59e0b; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>MONETA Financial Analyzer</h1>
                </div>
                <div class="content">
                    <h2>Password Reset Request</h2>
                    {"<p>Hello " + username + ",</p>" if username else "<p>Hello,</p>"}
                    <p>We received a request to reset your password for your MONETA Financial Analyzer account.</p>
                    <p>Click the button below to reset your password:</p>
                    <div style="text-align: center;">
                        <a href="{reset_link}" class="button">Reset Password</a>
                    </div>
                    <p>Or copy and paste this link into your browser:</p>
                    <p style="word-break: break-all; color: #2563eb;">{reset_link}</p>
                    <div class="warning">
                        <strong>⚠️ Important:</strong>
                        <ul>
                            <li>This link will expire in 1 hour</li>
                            <li>If you didn't request this reset, please ignore this email</li>
                            <li>For security, never share this link with anyone</li>
                        </ul>
                    </div>
                    <p>If you continue to have problems, please contact our support team.</p>
                    <p>Best regards,<br>The MONETA Team</p>
                </div>
                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text content
        text_content = f"""
MONETA Financial Analyzer - Password Reset Request

{"Hello " + username + "," if username else "Hello,"}

We received a request to reset your password for your MONETA Financial Analyzer account.

To reset your password, click the following link or copy and paste it into your browser:

{reset_link}

⚠️ Important:
- This link will expire in 1 hour
- If you didn't request this reset, please ignore this email
- For security, never share this link with anyone

If you continue to have problems, please contact our support team.

Best regards,
The MONETA Team

---
This is an automated message. Please do not reply to this email.
        """
        
        subject = "MONETA - Password Reset Request"
        return self.send_email(to_email, subject, html_content, text_content)
    
    def send_username_recovery_email(self, to_email: str, username: str) -> bool:
        """
        Send username recovery email
        
        Args:
            to_email: Recipient email address
            username: Username to send
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        # HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #2563eb; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
                .content {{ background-color: #f9fafb; padding: 30px; border-radius: 0 0 5px 5px; }}
                .username-box {{ background-color: white; padding: 15px; border-radius: 5px; border: 2px solid #2563eb; margin: 20px 0; text-align: center; }}
                .username {{ font-size: 24px; font-weight: bold; color: #2563eb; }}
                .footer {{ text-align: center; margin-top: 20px; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>MONETA Financial Analyzer</h1>
                </div>
                <div class="content">
                    <h2>Username Recovery</h2>
                    <p>Hello,</p>
                    <p>We received a request to recover your username for your MONETA Financial Analyzer account.</p>
                    <p>Your username is:</p>
                    <div class="username-box">
                        <div class="username">{username}</div>
                    </div>
                    <p>If you didn't request this information, please ignore this email or contact our support team if you have concerns.</p>
                    <p>Best regards,<br>The MONETA Team</p>
                </div>
                <div class="footer">
                    <p>This is an automated message. Please do not reply to this email.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Plain text content
        text_content = f"""
MONETA Financial Analyzer - Username Recovery

Hello,

We received a request to recover your username for your MONETA Financial Analyzer account.

Your username is:

{username}

If you didn't request this information, please ignore this email or contact our support team if you have concerns.

Best regards,
The MONETA Team

---
This is an automated message. Please do not reply to this email.
        """
        
        subject = "MONETA - Username Recovery"
        return self.send_email(to_email, subject, html_content, text_content)


# Create singleton instance
email_service = EmailService()

