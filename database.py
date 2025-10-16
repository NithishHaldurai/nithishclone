import sqlite3
import os
from datetime import datetime
import traceback

def init_db():
    """Initialize the database with required tables"""
    print("Initializing database...")
    try:
        conn = sqlite3.connect('clone_chat.db')
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute('PRAGMA foreign_keys = ON')
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✓ Users table created/verified")
        
        # Create messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                room TEXT DEFAULT 'general',
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        ''')
        print("✓ Messages table created/verified")
        
        # Create rooms table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rooms (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        print("✓ Rooms table created/verified")
        
        # Insert default rooms
        default_rooms = ['general', 'random', 'help']
        for room in default_rooms:
            cursor.execute('INSERT OR IGNORE INTO rooms (name) VALUES (?)', (room,))
        
        print("✓ Default rooms inserted")
        
        conn.commit()
        conn.close()
        print("✓ Database initialized successfully")
        
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        print(traceback.format_exc())

def get_db_connection():
    """Get database connection with error handling"""
    try:
        conn = sqlite3.connect('clone_chat.db')
        conn.row_factory = sqlite3.Row
        # Enable foreign keys
        conn.execute('PRAGMA foreign_keys = ON')
        return conn
    except Exception as e:
        print(f"✗ Error connecting to database: {e}")
        return None

def check_db_health():
    """Check if database and tables are properly set up"""
    try:
        conn = get_db_connection()
        if not conn:
            return False, "Failed to connect to database"
        
        cursor = conn.cursor()
        
        # Check if tables exist
        tables = ['users', 'messages', 'rooms']
        existing_tables = []
        
        for table in tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            if cursor.fetchone():
                existing_tables.append(table)
        
        conn.close()
        
        if len(existing_tables) == len(tables):
            return True, f"All tables exist: {existing_tables}"
        else:
            return False, f"Missing tables. Found: {existing_tables}"
            
    except Exception as e:
        return False, f"Database health check failed: {e}"

# User-related functions
def add_user(username, email=None):
    """Add a new user to the database"""
    print(f"Adding user: {username}")
    conn = get_db_connection()
    if not conn:
        return None
        
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO users (username, email) VALUES (?, ?)',
            (username, email)
        )
        conn.commit()
        user_id = cursor.lastrowid
        print(f"✓ User added with ID: {user_id}")
        return user_id
    except sqlite3.IntegrityError as e:
        print(f"✗ User already exists: {username}")
        return None
    except Exception as e:
        print(f"✗ Error adding user: {e}")
        return None
    finally:
        conn.close()

def get_user_by_username(username):
    """Get user by username"""
    conn = get_db_connection()
    if not conn:
        return None
        
    try:
        user = conn.execute(
            'SELECT * FROM users WHERE username = ?', (username,)
        ).fetchone()
        return user
    except Exception as e:
        print(f"✗ Error getting user {username}: {e}")
        return None
    finally:
        conn.close()

def get_all_users():
    """Get all users (for debugging)"""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        users = conn.execute('SELECT * FROM users ORDER BY username').fetchall()
        return users
    except Exception as e:
        print(f"✗ Error getting users: {e}")
        return []
    finally:
        conn.close()

# Message-related functions
def add_message(user_id, content, room='general'):
    """Add a new message to the database"""
    print(f"Adding message: user_id={user_id}, room={room}, content={content[:50]}...")
    conn = get_db_connection()
    if not conn:
        return None
        
    cursor = conn.cursor()
    try:
        cursor.execute(
            'INSERT INTO messages (user_id, content, room) VALUES (?, ?, ?)',
            (user_id, content, room)
        )
        conn.commit()
        message_id = cursor.lastrowid
        print(f"✓ Message added with ID: {message_id}")
        return message_id
    except Exception as e:
        print(f"✗ Error adding message: {e}")
        return None
    finally:
        conn.close()

def get_messages(room='general', limit=50):
    """Get messages for a room (newest first)"""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        messages = conn.execute('''
            SELECT m.*, u.username 
            FROM messages m 
            JOIN users u ON m.user_id = u.id 
            WHERE m.room = ? 
            ORDER BY m.timestamp DESC 
            LIMIT ?
        ''', (room, limit)).fetchall()
        print(f"✓ Retrieved {len(messages)} messages from room: {room}")
        return messages
    except Exception as e:
        print(f"✗ Error getting messages: {e}")
        return []
    finally:
        conn.close()

def get_recent_messages(room='general', limit=20):
    """Get recent messages in chronological order (oldest first)"""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        messages = conn.execute('''
            SELECT m.*, u.username 
            FROM messages m 
            JOIN users u ON m.user_id = u.id 
            WHERE m.room = ? 
            ORDER BY m.timestamp ASC 
            LIMIT ?
        ''', (room, limit)).fetchall()
        print(f"✓ Retrieved {len(messages)} recent messages from room: {room}")
        return messages
    except Exception as e:
        print(f"✗ Error getting recent messages: {e}")
        return []
    finally:
        conn.close()

def get_message_count():
    """Get total message count (for debugging)"""
    conn = get_db_connection()
    if not conn:
        return 0
        
    try:
        count = conn.execute('SELECT COUNT(*) as count FROM messages').fetchone()['count']
        return count
    except Exception as e:
        print(f"✗ Error getting message count: {e}")
        return 0
    finally:
        conn.close()

# Room-related functions
def get_rooms():
    """Get all available rooms"""
    conn = get_db_connection()
    if not conn:
        return []
        
    try:
        rooms = conn.execute('SELECT * FROM rooms ORDER BY name').fetchall()
        print(f"✓ Retrieved {len(rooms)} rooms")
        return rooms
    except Exception as e:
        print(f"✗ Error getting rooms: {e}")
        return []
    finally:
        conn.close()

def add_room(name):
    """Add a new room"""
    print(f"Adding room: {name}")
    conn = get_db_connection()
    if not conn:
        return False
        
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO rooms (name) VALUES (?)', (name,))
        conn.commit()
        print(f"✓ Room added: {name}")
        return True
    except sqlite3.IntegrityError:
        print(f"✗ Room already exists: {name}")
        return False
    except Exception as e:
        print(f"✗ Error adding room: {e}")
        return False
    finally:
        conn.close()

def get_room_messages_count(room='general'):
    """Get message count for a specific room"""
    conn = get_db_connection()
    if not conn:
        return 0
        
    try:
        count = conn.execute(
            'SELECT COUNT(*) as count FROM messages WHERE room = ?', 
            (room,)
        ).fetchone()['count']
        return count
    except Exception as e:
        print(f"✗ Error getting room message count: {e}")
        return 0
    finally:
        conn.close()

# Database maintenance functions
def clear_all_messages():
    """Clear all messages (for testing)"""
    conn = get_db_connection()
    if not conn:
        return False
        
    try:
        conn.execute('DELETE FROM messages')
        conn.commit()
        print("✓ All messages cleared")
        return True
    except Exception as e:
        print(f"✗ Error clearing messages: {e}")
        return False
    finally:
        conn.close()

def get_database_stats():
    """Get comprehensive database statistics"""
    try:
        stats = {
            'database_file_exists': os.path.exists('clone_chat.db'),
            'database_file_size': os.path.getsize('clone_chat.db') if os.path.exists('clone_chat.db') else 0,
            'users_count': len(get_all_users()),
            'messages_count': get_message_count(),
            'rooms_count': len(get_rooms()),
            'room_stats': {}
        }
        
        # Get message count per room
        rooms = get_rooms()
        for room in rooms:
            room_name = room['name']
            stats['room_stats'][room_name] = get_room_messages_count(room_name)
        
        return stats
    except Exception as e:
        print(f"✗ Error getting database stats: {e}")
        return {}

# Initialize database when this module is imported
if __name__ != '__main__':
    init_db()
    
    # Perform health check
    health_status, health_message = check_db_health()
    if health_status:
        print(f"✓ Database health check passed: {health_message}")
        
        # Print initial stats
        stats = get_database_stats()
        print("Database Statistics:")
        print(f"  - Users: {stats['users_count']}")
        print(f"  - Messages: {stats['messages_count']}")
        print(f"  - Rooms: {stats['rooms_count']}")
        for room, count in stats['room_stats'].items():
            print(f"  - {room}: {count} messages")
    else:
        print(f"✗ Database health check failed: {health_message}")

# Test function when run directly
if __name__ == '__main__':
    print("Testing database module...")
    init_db()
    
    # Test basic operations
    user_id = add_user("test_user")
    if user_id:
        add_message(user_id, "This is a test message!")
    
    stats = get_database_stats()
    print("\nFinal Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value}")