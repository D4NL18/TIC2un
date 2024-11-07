import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class CService {

  private apiUrl = 'http://localhost:5000/c';

  constructor(private http: HttpClient) { }

  trainC(): Observable<any> {
    return this.http.post(`${this.apiUrl}/run`, {});
  }

  getImage(): Observable<any> {
    return this.http.get(`${this.apiUrl}/image`, { responseType: 'blob' });
  }

}