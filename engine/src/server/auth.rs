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
    use super::{check_bearer, constant_time_eq};
    use crate::server::error::ServerError;
    use axum::http::{HeaderMap, HeaderValue, header};

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
