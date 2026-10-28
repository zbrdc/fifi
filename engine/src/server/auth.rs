//! Bearer-token check for the protected routes.
//!
//! Contract: `api/openapi.yaml` (`securitySchemes.apiKey`). A route with an empty `security`
//! array skips this check. The router applies it to the other routes.

use super::error::ServerError;
use axum::http::{HeaderMap, header};

/// Accepts the request only when `Authorization` carries `Bearer <key>`.
/// An empty key rejects every request, so a misconfigured server never runs open.
pub(crate) fn check_bearer(headers: &HeaderMap, key: &[u8]) -> Result<(), ServerError> {
    let value = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .ok_or(ServerError::Unauthorized)?;
    let (scheme, token) = value.split_once(' ').ok_or(ServerError::Unauthorized)?;
    if scheme.eq_ignore_ascii_case("bearer")
        && !key.is_empty()
        && constant_time_eq(token.as_bytes(), key)
    {
        Ok(())
    } else {
        Err(ServerError::Unauthorized)
    }
}

/// Accepts `/v1/messages`: `x-api-key: <key>` (the Anthropic client header) or the
/// `Authorization` bearer form. A set `x-api-key` must match — including one that
/// is not valid header text or one of several — with no fallback to the bearer
/// form, because a client that sends both uses one credential. An empty key
/// rejects every request.
pub(crate) fn check_messages(headers: &HeaderMap, key: &[u8]) -> Result<(), ServerError> {
    let api_keys = headers.get_all("x-api-key");
    if api_keys.iter().count() > 1 {
        return Err(ServerError::Unauthorized);
    }
    if let Some(value) = api_keys.iter().next() {
        return match value.to_str() {
            Ok(value) if !key.is_empty() && constant_time_eq(value.as_bytes(), key) => Ok(()),
            _ => Err(ServerError::Unauthorized),
        };
    }
    check_bearer(headers, key)
}

/// Compares two byte strings without an early exit on the first difference.
/// A length difference counts as a difference.
fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    let mut difference = left.len() ^ right.len();
    for index in 0..left.len().max(right.len()) {
        let a = left.get(index).copied().unwrap_or(0);
        let b = right.get(index).copied().unwrap_or(0);
        difference |= usize::from(a ^ b);
    }
    difference == 0
}

#[cfg(test)]
mod tests {
    use super::{check_bearer, check_messages, constant_time_eq};
    use crate::server::error::ServerError;
    use axum::http::{HeaderMap, HeaderValue, header};

    #[test]
    fn an_empty_configured_key_rejects_every_x_api_key_value() {
        let mut empty = HeaderMap::new();
        empty.insert("x-api-key", HeaderValue::from_static(""));
        assert!(matches!(
            check_messages(&empty, b""),
            Err(ServerError::Unauthorized)
        ));
        let mut wrong = HeaderMap::new();
        wrong.insert("x-api-key", HeaderValue::from_static("sentinel"));
        assert!(matches!(
            check_messages(&wrong, b""),
            Err(ServerError::Unauthorized)
        ));
    }

    fn headers(authorization: Option<&str>) -> HeaderMap {
        let mut map = HeaderMap::new();
        if let Some(value) = authorization {
            map.insert(
                header::AUTHORIZATION,
                HeaderValue::from_str(value).expect("header value"),
            );
        }
        map
    }

    #[test]
    fn the_configured_key_passes() {
        assert_eq!(
            check_bearer(&headers(Some("Bearer sk-local")), b"sk-local"),
            Ok(())
        );
    }

    #[test]
    fn the_messages_route_accepts_the_api_key_header() {
        let mut map = headers(None);
        map.insert(
            "x-api-key",
            HeaderValue::from_str("sk-local").expect("header value"),
        );
        assert_eq!(check_messages(&map, b"sk-local"), Ok(()));
    }

    #[test]
    fn the_messages_route_accepts_the_bearer_form() {
        assert_eq!(
            check_messages(&headers(Some("Bearer sk-local")), b"sk-local"),
            Ok(())
        );
    }

    #[test]
    fn a_wrong_api_key_header_is_unauthorized_without_bearer_fallback() {
        let mut map = headers(Some("Bearer sk-local"));
        map.insert(
            "x-api-key",
            HeaderValue::from_str("sk-other").expect("header value"),
        );
        assert_eq!(
            check_messages(&map, b"sk-local"),
            Err(ServerError::Unauthorized)
        );
    }

    #[test]
    fn the_messages_route_rejects_without_credentials() {
        assert_eq!(
            check_messages(&headers(None), b"sk-local"),
            Err(ServerError::Unauthorized)
        );
    }

    #[test]
    fn an_api_key_that_is_not_header_text_is_unauthorized() {
        let mut map = headers(Some("Bearer sk-local"));
        map.insert(
            "x-api-key",
            HeaderValue::from_bytes(&[0xff, 0xfe]).expect("opaque header value"),
        );
        assert_eq!(
            check_messages(&map, b"sk-local"),
            Err(ServerError::Unauthorized)
        );
    }

    #[test]
    fn two_api_key_headers_are_unauthorized() {
        let mut map = headers(None);
        map.append(
            "x-api-key",
            HeaderValue::from_str("sk-local").expect("header value"),
        );
        map.append(
            "x-api-key",
            HeaderValue::from_str("sk-local").expect("header value"),
        );
        assert_eq!(
            check_messages(&map, b"sk-local"),
            Err(ServerError::Unauthorized)
        );
    }

    #[test]
    fn the_scheme_name_ignores_case() {
        assert_eq!(
            check_bearer(&headers(Some("bearer sk-local")), b"sk-local"),
            Ok(())
        );
    }

    #[test]
    fn a_missing_header_is_unauthorized() {
        assert_eq!(
            check_bearer(&headers(None), b"sk-local"),
            Err(ServerError::Unauthorized)
        );
    }

    #[test]
    fn a_wrong_token_is_unauthorized() {
        assert_eq!(
            check_bearer(&headers(Some("Bearer sk-other")), b"sk-local"),
            Err(ServerError::Unauthorized)
        );
    }

    #[test]
    fn another_scheme_with_the_right_token_is_unauthorized() {
        assert_eq!(
            check_bearer(&headers(Some("Basic sk-local")), b"sk-local"),
            Err(ServerError::Unauthorized)
        );
    }

    #[test]
    fn a_header_without_a_space_is_unauthorized() {
        assert_eq!(
            check_bearer(&headers(Some("sk-local")), b"sk-local"),
            Err(ServerError::Unauthorized)
        );
    }

    #[test]
    fn an_empty_key_rejects_an_empty_token() {
        assert_eq!(
            check_bearer(&headers(Some("Bearer ")), b""),
            Err(ServerError::Unauthorized)
        );
    }

    #[test]
    fn equal_strings_compare_equal() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(constant_time_eq(b"", b""));
    }

    #[test]
    fn a_difference_in_the_last_byte_is_found() {
        assert!(!constant_time_eq(b"abc", b"abd"));
    }

    #[test]
    fn a_difference_in_the_first_byte_is_found() {
        assert!(!constant_time_eq(b"xbc", b"abc"));
    }

    #[test]
    fn a_zero_padded_string_differs_from_the_original() {
        assert!(!constant_time_eq(b"ab", b"ab\0"));
        assert!(!constant_time_eq(b"ab\0", b"ab"));
    }
}
