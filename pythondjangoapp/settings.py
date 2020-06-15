
import os


DEBUG = True

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.11/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!

SECRET_KEY = '+)!83krz^m$i%=uo!4b7ps2s7q=2ikcm#sw!c=#6v9er#hx8or'


ALLOWED_HOSTS = ['*']


# Application definition
INSTALLED_APPS = [
                  'django.contrib.admin',
                  'django.contrib.auth',
                  'django.contrib.contenttypes',
                  'django.contrib.sessions',
                  'django.contrib.messages',
                  'django.contrib.staticfiles',
                  'livereload',
                  'app',
                  ]

MIDDLEWARE = [
              'django.middleware.security.SecurityMiddleware',
              'whitenoise.middleware.WhiteNoiseMiddleware',
              'django.contrib.sessions.middleware.SessionMiddleware',
              'django.middleware.common.CommonMiddleware',
              'django.middleware.csrf.CsrfViewMiddleware',
              'django.contrib.auth.middleware.AuthenticationMiddleware',
              'django.contrib.messages.middleware.MessageMiddleware',
              'django.middleware.clickjacking.XFrameOptionsMiddleware',
              ]


ROOT_URLCONF = 'pythondjangoapp.urls'

TEMPLATES = [
             {
             'BACKEND': 'django.template.backends.django.DjangoTemplates',
             'DIRS': [os.path.join(BASE_DIR, 'app', 'templates')],
             'APP_DIRS': True,
             'OPTIONS': {
             'context_processors': [
                                    'django.template.context_processors.debug',
                                    'django.template.context_processors.request',
                                    'django.contrib.auth.context_processors.auth',
                                    'django.contrib.messages.context_processors.messages',
                                    ],
             },
             },
             ]

WSGI_APPLICATION = 'pythondjangoapp.wsgi.application'

# Password validation
# https://docs.djangoproject.com/en/1.11/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
                            {
                            'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
                            },
                            {
                            'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
                            },
                            {
                            'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
                            },
                            {
                            'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
                            },
                            ]

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/1.11/howto/static-files/
STATIC_URL = '/static/'

STATICFILES_DIRS = (
                    os.path.join(BASE_DIR, "app", "static"),
                    os.path.join(BASE_DIR, "staticfile"),
                    )

STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(BASE_DIR, 'db.sqlite3'),
    }
}

# CHANNEL_LAYERS = {
#     'default': {
#         'BACKEND': 'channels_redis.core.RedisChannelLayer',
#         'CONFIG': {
#             'hosts': [("redis://:6918xm1KN07HWlcEIAM1tZhKkLBGC804@redis-19049.c61.us-east-1-3.ec2.cloud.redislabs.com:19049")],
#         },
#         'ROUTING': "pythondjangoapp.routing.application",
#     }
# }

LOGIN_URL = "/login"
LOGOUT_REDIRECT_URL = "/"
